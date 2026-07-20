"""Data loading, saving, and normalisation for beam-NODE experiments."""
from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def load_dataset(
    path: str,
    normalize: bool = False,
    disp_norm: float | None = None,
    vel_norm: float | None = None,
    ft_norm: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, float, float, NDArray]:
    """Load snapshot, parameter, and forcing arrays from *path*.

    Uses memory-mapped loading (``mmap_mode="c"``) so only accessed pages
    are read into RAM — enables processing datasets larger than physical
    memory.

    Parameters
    ----------
    path : str
        Directory containing ``snapshot_data.npy``, ``params.npy``, ``forcing.npy``.
    normalize : bool
        If True, normalise displacements, velocities, and forcing.
    disp_norm, vel_norm : float or None
        Pre-computed norms.  If None, computed from the data.
    ft_norm : NDArray or None
        Pre-computed forcing normalisation array ``[n_cols, 2]``.

    Returns
    -------
    snapshot_data, parameters, forcing, disp_norm, vel_norm, ft_norm
    """
    # Memory-mapped, copy-on-write — pages loaded on demand
    snapshot_data = np.load(os.path.join(path, "snapshot_data.npy"), mmap_mode="c")
    parameters = np.load(os.path.join(path, "params.npy"), mmap_mode="c")
    forcing = np.load(os.path.join(path, "forcing.npy"), mmap_mode="c")

    # If we need to mutate, make writable copies
    if normalize:
        snapshot_data = np.array(snapshot_data)
        forcing = np.array(forcing)

    if disp_norm is not None and normalize:
        snapshot_data[:, 0] /= disp_norm
        snapshot_data[:, 1] /= vel_norm
    else:
        disp_norm = float(np.max(np.abs(snapshot_data[:, 0])))
        vel_norm = float(np.max(np.abs(snapshot_data[:, 1])))
        if normalize:
            snapshot_data[:, 0] /= disp_norm
            snapshot_data[:, 1] /= vel_norm

    if ft_norm is not None and normalize:
        for col in range(forcing.shape[-1]):
            forcing[:, :, col] = (forcing[:, :, col] - ft_norm[col, 1]) / (ft_norm[col, 0] - ft_norm[col, 1])
    else:
        ft_norm = np.zeros([forcing.shape[-1], 2])
        for col in range(forcing.shape[-1]):
            ft_norm[col, 0] = np.max(forcing[:, :, col])
            ft_norm[col, 1] = np.min(forcing[:, :, col])
            if normalize:
                forcing[:, :, col] = (forcing[:, :, col] - ft_norm[col, 1]) / (ft_norm[col, 0] - ft_norm[col, 1])

    return snapshot_data, parameters, forcing, disp_norm, vel_norm, ft_norm


def save_dataset(
    path: str,
    data: NDArray,
    params: NDArray,
    forcing: NDArray,
    cluster: int | None = None,
    disp_norm: float | None = None,
    vel_norm: float | None = None,
    ft_norm: NDArray | None = None,
) -> None:
    """Save snapshot, parameter, and forcing arrays to *path*."""
    if disp_norm is not None:
        data[:, 0] *= disp_norm
        data[:, 1] *= vel_norm
        for col in range(forcing.shape[-1]):
            forcing[:, :, col] *= ft_norm[col, 0] - ft_norm[col, 1]
            forcing[:, :, col] += ft_norm[col, 1]

    if cluster is not None:
        os.makedirs(os.path.join(path, f"{cluster}"), exist_ok=True)
        np.save(os.path.join(path, f"{cluster}/snapshot_data.npy"), data)
        np.save(os.path.join(path, f"{cluster}/params.npy"), params)
        np.save(os.path.join(path, f"{cluster}/forcing.npy"), forcing)
    else:
        np.save(os.path.join(path, "snapshot_data.npy"), data)
        np.save(os.path.join(path, "params.npy"), params)
        np.save(os.path.join(path, "forcing.npy"), forcing)


def load_cluster(
    path: str,
    cluster: int,
    normalize: bool = False,
    disp_norm: float | None = None,
    vel_norm: float | None = None,
    ft_norm: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, float, float, NDArray]:
    """Load a clustered sub-dataset."""
    return load_dataset(os.path.join(path, str(cluster)), normalize, disp_norm, vel_norm, ft_norm)


def save_cluster(
    path: str,
    cluster: int,
    data: NDArray,
    params: NDArray,
    forcing: NDArray,
    disp_norm: float | None = None,
    vel_norm: float | None = None,
    ft_norm: NDArray | None = None,
) -> None:
    """Save a clustered sub-dataset."""
    os.makedirs(os.path.join(path, str(cluster)), exist_ok=True)
    save_dataset(path, data, params, forcing, cluster, disp_norm, vel_norm, ft_norm)


def parameter_plot(
    param_train: NDArray,
    param_val: NDArray,
    param_test: NDArray,
) -> None:
    """Plot parameter distributions for train/val/test splits."""
    plt.figure(figsize=(6, 4))

    if param_train.shape[1] == 1:
        x_train = np.zeros(param_train.shape[0])
        x_val = np.ones(param_val.shape[0])
        x_test = np.full(param_test.shape[0], 2)

        plt.scatter(x_train, param_train, label="Train", alpha=0.6)
        plt.scatter(x_val, param_val, label="Val", alpha=0.6)
        plt.scatter(x_test, param_test, label="Test", alpha=0.6)

        plt.xticks([0, 1, 2], ["Train", "Val", "Test"])
        plt.ylabel("Parameter Value")
        plt.title("Parameter Distribution")
        plt.legend()
        plt.grid(True)

    elif param_train.shape[1] == 2:
        plt.scatter(param_train[:, 0], param_train[:, 1], label="Train", alpha=0.6)
        plt.scatter(param_val[:, 0], param_val[:, 1], label="Val", alpha=0.6)
        plt.scatter(param_test[:, 0], param_test[:, 1], label="Test", alpha=0.6)

        plt.xlabel("Parameter 1")
        plt.ylabel("Parameter 2")
        plt.title("Parameter Distribution")
        plt.legend()
        plt.grid(True)
    else:
        print("Only supports 1D/2D parameters for now.")
    plt.show()
