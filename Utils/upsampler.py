
import numpy as np
import matplotlib.pyplot as plt

def fourier_upsample_add(signal, add_points):
    """
    Fourier-based upsampling of a one-dimensional real signal via zero-padding.

    This method computes the discrete Fourier transform (DFT) of the input signal,
    pads the Fourier coefficients with zeros to reach a new length (original length + add_points),
    and computes the inverse DFT of the padded spectrum. For even-length signals, the Nyquist frequency
    coefficient is split equally between two bins to preserve conjugate symmetry. The final result
    is scaled by (new_length / original_length) to preserve amplitude.

    Parameters
    ----------
    signal : array_like
        Input one-dimensional real signal.
    add_points : int
        Number of additional points to be added. The new signal length is len(signal) + add_points.

    Returns
    -------
    upsampled_signal : ndarray
        The upsampled signal with enhanced resolution.

    Raises
    ------
    ValueError
        If add_points is negative or not an integer.
    """
    if add_points < 0 or not isinstance(add_points, int):
        raise ValueError("The additional points must be a non-negative integer.")

    N = len(signal)
    M = N + add_points  # new signal length
    X = np.fft.fft(signal)
    Y = np.zeros(M, dtype=complex)

    if N % 2 == 0:
        # For an even-length signal.
        k = N // 2
        # Copy lower frequencies.
        Y[:k] = X[:k]
        # Split the Nyquist frequency:
        Y[k] = X[k] / 2
        Y[M - k] = X[k] / 2
        # The midsection remains zero (padding).
        # Copy the remaining high frequencies.
        Y[M - k + 1:] = X[k+1:]
    else:
        # For an odd-length signal.
        k = (N + 1) // 2
        # Copy lower frequencies.
        Y[:k] = X[:k]
        # Copy the remaining high frequencies.
        Y[M - (N - k):] = X[k:]

    # Multiply by scaling factor (M/N) to preserve amplitude.
    upsampled_signal = np.fft.ifft(Y) * (M / N)
    return np.real(upsampled_signal)
