"""
VTK export utilities.

This module provides functions for:
- plotting a 2D field on a scikit-fem basis
- writing one displacement vector to a VTK file, with mesh translation
- exporting paired full-order and reduced-order solutions to VTK
- exporting a displacement time series to VTK with a PVD collection file
- exporting a point-data time series to VTK on an undeformed mesh

Notes
-----
Authors: Suparno Bhattacharyya
"""

import os
import shutil
from pathlib import Path

import numpy as np


def visualize2D(u, basis):
    """
    Plot a scalar field on a 2D scikit-fem basis.

    Parameters
    ----------
    u : array_like
        Field values compatible with the given basis.
    basis : object
        scikit-fem basis.

    Returns
    -------
    out
        Return value from the backend plot/show call.
    """
    from skfem.visuals.matplotlib import plot

    return plot(
        basis,
        u,
        shading="gouraud",
        colorbar="True",
    ).show()


def save_vtk_solution(
    u,
    mesh,
    basis,
    scale,
    run_dir,
    prefix,
    split_dim: bool = False,
):
    """
    Write one solution vector to a VTK file.

    The function extracts nodal degrees of freedom from ``u`` using
    ``basis.nodal_dofs``, scales the nodal values by ``scale``, translates the
    mesh, and writes one ``.vtk`` file. If ``split_dim`` is True, the nodal
    displacement components are stored as scalar point-data fields.

    Parameters
    ----------
    u : array_like
        Global solution vector that contains nodal degrees of freedom indexed by
        ``basis.nodal_dofs``.
    mesh : object
        Mesh object that provides:
        - ``translated(u_node)`` returning a new mesh
        - ``save(path, point_data=...)`` writing a VTK file
    basis : object
        Basis object that provides ``nodal_dofs``.
    scale : float
        Scale factor applied to nodal values before translation.
    run_dir : str or pathlib.Path
        Output directory.
    prefix : str
        Output file prefix. The file name is ``{prefix}.vtk``.
    split_dim : bool, optional
        If True, write scalar fields ``u_x``, ``u_y``, ``u_z`` (up to the mesh
        dimension). If False, write only the displaced mesh. Default is False.

    Returns
    -------
    None

    Notes
    -----
    Authors: Suparno Bhattacharyya
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    u = np.asarray(u)

    # Extract nodal dofs and scale.
    # For vector problems, basis.nodal_dofs often has shape (dim, n_nodes).
    u_node = scale * u[basis.nodal_dofs]

    # Translate the mesh by nodal displacement.
    m = mesh.translated(u_node)

    filename = run_dir / f"{prefix}.vtk"

    if not split_dim:
        m.save(str(filename))
        return

    point_data = {}

    if u_node.ndim == 1:
        # Scalar fallback
        point_data["u"] = u_node
    else:
        dim, _ = u_node.shape
        labels = ["x", "y", "z"][:dim]
        for d, lab in enumerate(labels):
            point_data[f"u_{lab}"] = u_node[d, :]

    m.save(str(filename), point_data=point_data)


def generate_vtk(
    LS_test,
    LS_rom,
    mesh,
    basis,
    scale: float = 1.0,
    num_test: int = 5,
    out_dir: str = "sol_vtk_files",
    split_dim: bool = False,
    clean_out_dir: bool = False,
):
    """
    Export paired full-order and reduced-order solutions to VTK.

    The function selects ``num_test`` random indices and writes one full-order
    and one reduced-order VTK file in a subfolder ``Test_i``.

    Parameters
    ----------
    LS_test : sequence of array_like
        Full-order solution vectors.
    LS_rom : sequence of array_like
        Reduced-order solution vectors aligned with LS_test by index.
    mesh : object
        Mesh object passed to ``save_vtk_solution``.
    basis : object
        Basis object passed to ``save_vtk_solution``.
    scale : float, optional
        Scale factor for nodal values. Default is 1.0.
    num_test : int, optional
        Number of cases to export. Default is 5.
    out_dir : str, optional
        Output directory. Subfolders ``Test_1``, ``Test_2``, ... are created.
        Default is "sol_vtk_files".
    split_dim : bool, optional
        Passed to ``save_vtk_solution``. Default is False.
    clean_out_dir : bool, optional
        If True, remove ``out_dir`` before writing. Default is False.

    Returns
    -------
    None

    Notes
    -----
    Authors: Suparno Bhattacharyya
    """
    base_dir = Path(out_dir)

    if clean_out_dir and base_dir.exists():
        shutil.rmtree(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)

    n = len(LS_test)
    if len(LS_rom) != n:
        raise ValueError("LS_test and LS_rom must have the same length.")

    for i in range(1, num_test + 1):
        idx = np.random.randint(n)
        run_dir = base_dir / f"Test_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)

        save_vtk_solution(
            LS_test[idx],
            mesh,
            basis,
            scale,
            run_dir,
            prefix=f"test_sol_fos_{i}",
            split_dim=split_dim,
        )

        save_vtk_solution(
            LS_rom[idx],
            mesh,
            basis,
            scale,
            run_dir,
            prefix=f"test_sol_rom_{i}",
            split_dim=split_dim,
        )


def save_vtk_time_series(
    U: np.ndarray,
    times: np.ndarray,
    mesh,
    basis,
    scale: float,
    run_dir: Path,
    prefix: str,
    interval: int = 10,
):
    """
    Write one VTK per saved time step and write a PVD collection file.

    The function saves frames named ``{prefix}_{k:04d}.vtk`` and writes a PVD
    file named ``{prefix}.pvd`` that references the frames and their times.

    Parameters
    ----------
    U : numpy.ndarray
        Displacement history with shape (ndofs, n_steps).
    times : numpy.ndarray
        Time values with shape (n_steps,).
    mesh : object
        Mesh object passed to ``save_vtk_solution``.
    basis : object
        Basis object passed to ``save_vtk_solution``.
    scale : float
        Scale factor for nodal values before translation.
    run_dir : pathlib.Path
        Output directory.
    prefix : str
        Base name used for VTK frames and the PVD file.
    interval : int, optional
        Save every ``interval`` steps. Default is 10.

    Returns
    -------
    None

    Notes
    -----
    Authors: Suparno Bhattacharyya
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times)
    if U.shape[1] != times.shape[0]:
        raise ValueError("U must have shape (ndofs, n_steps) with n_steps == len(times).")

    pvd_file = run_dir / f"{prefix}.pvd"

    with pvd_file.open("w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")

        step_ids = range(0, times.shape[0], interval)
        for out_i, step in enumerate(step_ids):
            u = U[:, step]
            t = times[step]
            frame = f"{prefix}_{out_i:04d}"

            save_vtk_solution(u, mesh, basis, scale, run_dir, frame)

            f.write(
                f'    <DataSet timestep="{t}" group="" part="0" file="{frame}.vtk"/>\n'
            )

        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


def save_vtk_time_series_point(
    U: np.ndarray,
    mesh,
    run_dir: Path,
    prefix: str,
    interval: int = 10,
    point_data_name: str = "Temperature",
):
    """
    Write point-data frames to VTK on an undeformed mesh.

    The function writes frames named ``{prefix}_{k:04d}.vtk`` with a point-data
    array stored under ``point_data_name``.

    The time axis handling is:
    - if ``U.shape[0] == n_points``: treat U as (n_points, n_steps) and save columns
    - else: treat U as (n_steps, n_points) and save rows

    Parameters
    ----------
    U : numpy.ndarray
        Point-data history with shape (n_steps, n_points) or (n_points, n_steps).
    mesh : object
        Mesh object that provides ``save(path, point_data=...)``.
    run_dir : pathlib.Path
        Output directory.
    prefix : str
        Base name used for VTK frames.
    interval : int, optional
        Save every ``interval`` steps. Default is 10.
    point_data_name : str, optional
        Key used in VTK point_data. Default is "Temperature".

    Returns
    -------
    None

    Notes
    -----
    Authors: Suparno Bhattacharyya
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    U = np.asarray(U)
    n_points = mesh.p.shape[1] if hasattr(mesh, "p") else None

    if n_points is not None and U.ndim == 2 and U.shape[0] == n_points:
        # U: (n_points, n_steps)
        step_ids = range(0, U.shape[1], interval)
        for out_i, step in enumerate(step_ids):
            frame = f"{prefix}_{out_i:04d}"
            mesh.save(run_dir / f"{frame}.vtk", point_data={point_data_name: U[:, step]})
    else:
        # U: (n_steps, n_points)
        step_ids = range(0, U.shape[0], interval)
        for out_i, step in enumerate(step_ids):
            frame = f"{prefix}_{out_i:04d}"
            mesh.save(run_dir / f"{frame}.vtk", point_data={point_data_name: U[step, :]})


def save_vtu_time_series_point(
    U: np.ndarray,
    mesh,
    run_dir: Path,
    prefix: str,
    interval: int = 10,
    point_data_name: str = "Temperature",
):
    """
    Write point-data frames to VTK on an undeformed mesh.

    The function writes frames named ``{prefix}_{k:04d}.vtk`` with a point-data
    array stored under ``point_data_name``.

    The time axis handling is:
    - if ``U.shape[0] == n_points``: treat U as (n_points, n_steps) and save columns
    - else: treat U as (n_steps, n_points) and save rows

    Parameters
    ----------
    U : numpy.ndarray
        Point-data history with shape (n_steps, n_points) or (n_points, n_steps).
    mesh : object
        Mesh object that provides ``save(path, point_data=...)``.
    run_dir : pathlib.Path
        Output directory.
    prefix : str
        Base name used for VTU frames.
    interval : int, optional
        Save every ``interval`` steps. Default is 10.
    point_data_name : str, optional
        Key used in VTK point_data. Default is "Temperature".

    Returns
    -------
    None

    Notes
    -----
    Authors: Suparno Bhattacharyya
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    U = np.asarray(U)
    n_points = mesh.p.shape[1] if hasattr(mesh, "p") else None

    if n_points is not None and U.ndim == 2 and U.shape[0] == n_points:
        # U: (n_points, n_steps)
        step_ids = range(0, U.shape[1], interval)
        for out_i, step in enumerate(step_ids):
            frame = f"{prefix}_{out_i:04d}"
            mesh.save(run_dir / f"{frame}.vtu", point_data={point_data_name: U[:, step]})
    else:
        # U: (n_steps, n_points)
        step_ids = range(0, U.shape[0], interval)
        for out_i, step in enumerate(step_ids):
            frame = f"{prefix}_{out_i:04d}"
            mesh.save(run_dir / f"{frame}.vtu", point_data={point_data_name: U[step, :]})


def save_vtu_time_series_point_vertexsample(U, mesh, basis, run_dir, prefix, interval=8, point_data_name="Temperature"):
    """
    U: (n_steps, ndofs_P2)
    Saves vertex-sampled field (n_vertices,) on the original mesh.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    P = basis.probes(mesh.p)                     # maps P2 dofs -> values at vertices (n_vertices x ndofs)

    step_ids = np.arange(0, U.shape[0], interval)
    for out_i, step in enumerate(step_ids):
        frame = f"{prefix}_{out_i:04d}"
        u_vertex = P @ U[step, :]                # (n_vertices,)
        mesh.save(run_dir / f"{frame}.vtu", point_data={point_data_name: u_vertex})