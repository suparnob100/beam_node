"""Fourier-based signal upsampling via zero-padding in the frequency domain."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fourier_upsample_add(signal: NDArray, add_points: int) -> NDArray:
    """Upsample a 1-D real signal by zero-padding its DFT.

    For even-length signals the Nyquist coefficient is split equally
    between two bins to preserve conjugate symmetry.  The result is
    scaled by ``M / N`` to preserve amplitude.

    Parameters
    ----------
    signal : NDArray
        Input 1-D real signal.
    add_points : int
        Number of additional points.  New length = ``len(signal) + add_points``.

    Returns
    -------
    NDArray
        Upsampled signal (real part only).

    Raises
    ------
    ValueError
        If *add_points* is negative or not an integer.
    """
    if add_points < 0 or not isinstance(add_points, int):
        raise ValueError("The additional points must be a non-negative integer.")

    N = len(signal)
    M = N + add_points
    X = np.fft.fft(signal)
    Y = np.zeros(M, dtype=complex)

    if N % 2 == 0:
        k = N // 2
        Y[:k] = X[:k]
        Y[k] = X[k] / 2
        Y[M - k] = X[k] / 2
        Y[M - k + 1 :] = X[k + 1 :]
    else:
        k = (N + 1) // 2
        Y[:k] = X[:k]
        Y[M - (N - k) :] = X[k:]

    return np.real(np.fft.ifft(Y) * (M / N))
