"""Power spectral density computation with decimation-based anti-aliasing."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.signal.windows import hann


def psd_custom(t: NDArray, ts: NDArray) -> tuple[NDArray, NDArray]:
    """Compute PSD using Welch's method over the full signal.

    Parameters
    ----------
    t : NDArray
        Uniform time vector.
    ts : NDArray
        Time-series signal.

    Returns
    -------
    F : NDArray
        Frequency vector.
    Pxx : NDArray
        PSD in dB (10 * log10).
    """
    dt = t[1] - t[0]
    fs = 1.0 / dt

    nblock = len(ts)
    overlap = 1024
    win = hann(nblock)

    F, Pxx = welch(x=ts, fs=fs, window=win, noverlap=overlap, nfft=nblock, detrend=False, return_onesided=True)
    Pxx = 10.0 * np.log10(Pxx)

    return F, Pxx


def psd_cutoff(t: NDArray, ts: NDArray, f_cut: float) -> tuple[NDArray, NDArray]:
    """Compute PSD with decimation-based frequency cut-off.

    The signal is decimated by factor *m* (derived from ``f_cut`` and the
    sampling rate).  Each decimation phase is passed through Welch's method
    and the results are averaged.

    Parameters
    ----------
    t : NDArray
        Uniform time vector.
    ts : NDArray
        Time-series signal.
    f_cut : float
        Cut-off frequency (Hz).

    Returns
    -------
    F : NDArray
        Frequency vector.
    Pxx : NDArray
        PSD in dB (10 * log10).
    """
    dt = t[1] - t[0]
    fs = 1.0 / dt

    # Derive t_end from the time array (NOT a global variable)
    t_end = t[-1]

    m = int(np.floor(0.5 * fs / f_cut))
    lenC = int(np.floor(len(ts) / m))

    tn = np.linspace(0, t_end, num=lenC, endpoint=False)
    dtn = tn[-1] - tn[-2]
    fsn = 1.0 / dtn

    # Accumulate PSD estimates in a list (avoid repeated np.vstack)
    psd_list: list[NDArray] = []

    for loop in range(m):
        tmpvar = ts[loop::m]
        nblock = lenC
        overlap = 0
        win = hann(nblock)
        F, Px = welch(x=tmpvar[:lenC], fs=fsn, window=win, noverlap=overlap, nfft=nblock, detrend=False, return_onesided=True)
        psd_list.append(Px)

    if m > 1:
        Pxx = np.mean(np.vstack(psd_list), axis=0)
    else:
        Pxx = psd_list[0]

    Pxx = 10.0 * np.log10(Pxx)

    return F, Pxx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt = 0.01
    fs = 1.0 / dt
    t_end = 10000

    t = np.linspace(0, t_end, num=int(t_end * fs), endpoint=False)
    y = np.sin(2 * np.pi * 5 * t)

    F, Pxx = psd_cutoff(t, y, 10)
    plt.plot(F, Pxx)
    plt.show()
