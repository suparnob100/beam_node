"""Sensor placement, SVD-based compression, and sparse measurement utilities.

Backward compatible with the original scalar-channel workflow, and extended to
support grouped node-wise sensing for vector-valued FEA fields with interleaved DOFs:
    [ux0, uy0, ux1, uy1, ..., uxn, uyn]

Supported modes
---------------
1) sensor_mode = "scalar"       -> old behavior
   Candidate sensors are the last-axis scalar channels.

2) sensor_mode = "vector_nodes" -> new behavior
   Candidate sensors are physical nodes.
   Each selected node measures all components at that node.
   For interleaved 2D displacement:
       node i -> [ux_i, uy_i]

Minimal config examples
-----------------------
# Old 1D / scalar case
config = {
    "seed": 0,
    "sensors": {
        "n_SVD_basis": 20,
        "n_A_basis": 20,
        "sensor_mode": "scalar",   # optional; default is scalar
    },
    "data": {
        "nx": 101
    }
}

# New vector-node case for [ux0, uy0, ux1, uy1, ...]
config = {
    "seed": 0,
    "sensors": {
        "n_SVD_basis": 30,
        "n_A_basis": 30,
        "sensor_mode": "vector_nodes",
        "n_components": 2,
        "dof_layout": "interleaved",   # "interleaved" or "blocked"
    },
    "data": {
        "n_nodes": 101,
        # optional geometry-aware gap filling:
        # "node_coords": coords_array_of_shape_[n_nodes, dim]
    }
}
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from numpy.typing import NDArray
import numpy.linalg as LA
import pysensors as ps
import matplotlib.pyplot as plt


class sensor_processing:
    """Optimal sensor placement and compression matrix computation.

    Parameters
    ----------
    data : NDArray
        Snapshot data.

        Scalar mode:
            Any array whose last axis is the candidate sensor/channel axis.
            Examples:
                [n_params, n_fields, n_t, nx]
                [num_samples, time_steps, nx]

        Vector-node mode:
            Any array whose last axis is the full interleaved/blocked DOF axis.
            Examples:
                [num_samples, time_steps, dof]
                [n_cases, n_t, dof]
            where
                dof = n_nodes * n_components
    config : dict
        Configuration dictionary.
    """

    def __init__(self, data: NDArray, config: dict) -> None:
        self.data = np.asarray(data)
        self.config = config

        self.seed = config["seed"]
        self.n_SVD_basis = config["sensors"]["n_SVD_basis"]
        self.n_A_basis = config["sensors"]["n_A_basis"]
        self.err_cap = config["sensors"].get("err_cap", 2e-5)

        sensors_cfg = config.get("sensors", {})
        data_cfg = config.get("data", {})

        # Backward-compatible default: scalar
        self.sensor_mode = sensors_cfg.get("sensor_mode", "scalar")
        if self.sensor_mode not in {"scalar", "vector_nodes"}:
            raise ValueError(
                f"Unsupported sensor_mode={self.sensor_mode!r}. "
                f"Use 'scalar' or 'vector_nodes'."
            )

        self.n_components = int(sensors_cfg.get("n_components", 2))
        self.dof_layout = sensors_cfg.get("dof_layout", "blocked")
        if self.dof_layout not in {"interleaved", "blocked"}:
            raise ValueError(
                f"Unsupported dof_layout={self.dof_layout!r}. "
                f"Use 'interleaved' or 'blocked'."
            )

        # Optional geometry for gap filling in vector mode.
        node_coords = data_cfg.get("node_coords", None)
        self.node_coords = None if node_coords is None else np.asarray(node_coords)

        # Internal placeholders
        self.ps_model = None
        self.sensor_placement = None          # scalar indices OR node indices
        self.sensor_dof_indices = None        # only used in vector_nodes mode
        self.C_Mat = None
        self.A_Mat = None
        self.pinv_Theta = None
        self.s_count = None
        self.u = None
        self.s = None
        self.v = None

        # Prepare snapshot matrices
        if self.sensor_mode == "scalar":
            self.nx = int(data_cfg.get("nx", self.data.shape[-1]))
            if self.data.shape[-1] != self.nx:
                print("\033[93m"
                    f"Scalar mode expects last axis = nx. "
                    f"Got data.shape[-1]={self.data.shape[-1]} and nx={self.nx}."
                    "\033[0m"
                )

            # Backward-compatible: flatten all leading dims into snapshots
            self.data_reshaped = self.data.reshape(-1, self.nx)

            # Selection and reconstruction are the same in scalar mode
            self.selection_snapshot_matrix = self.data_reshaped
            self.full_snapshot_matrix = self.data_reshaped
            self.n_candidate_locations = self.nx

        else:  # vector_nodes
            dof = self.data.shape[-1]
            if dof % self.n_components != 0:
                raise ValueError(
                    f"Vector-node mode expects dof divisible by n_components. "
                    f"Got dof={dof}, n_components={self.n_components}."
                )

            self.n_nodes = int(data_cfg.get("n_nodes", dof // self.n_components))
            if dof != self.n_nodes * self.n_components:
                raise ValueError(
                    f"Inconsistent n_nodes and n_components with data dof. "
                    f"Got dof={dof}, n_nodes={self.n_nodes}, "
                    f"n_components={self.n_components}."
                )

            if self.node_coords is not None and len(self.node_coords) != self.n_nodes:
                raise ValueError(
                    f"node_coords length must equal n_nodes. "
                    f"Got len(node_coords)={len(self.node_coords)}, n_nodes={self.n_nodes}."
                )

            # Full-state snapshots for reconstruction basis
            self.data_reshaped = self.data.reshape(-1, dof)
            self.full_snapshot_matrix = self.data_reshaped

            # Node-wise selection snapshots:
            # each column corresponds to a node, and all components contribute
            self.selection_snapshot_matrix = self._build_node_selection_snapshot_matrix(
                self.data_reshaped,
                n_nodes=self.n_nodes,
                n_components=self.n_components,
                dof_layout=self.dof_layout,
            )
            self.n_candidate_locations = self.n_nodes

            # Keep legacy name available for any downstream code
            self.nx = self.n_nodes

    @staticmethod
    def _safe_normalize(X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        scale = np.max(np.abs(X))
        if scale == 0.0:
            return X.copy()
        return X / scale

    @staticmethod
    def _build_node_selection_snapshot_matrix(
        X_full: NDArray,
        n_nodes: int,
        n_components: int,
        dof_layout: str,
    ) -> NDArray:
        """Build a snapshot matrix whose columns correspond to nodes.

        Parameters
        ----------
        X_full : [n_snapshots, dof]
            Full-state snapshot matrix.
        n_nodes : int
            Number of physical nodes.
        n_components : int
            Number of components per node.
        dof_layout : str
            "interleaved" -> [ux0, uy0, ux1, uy1, ...]
            "blocked"     -> [ux_all_nodes, uy_all_nodes, ...]

        Returns
        -------
        X_nodes : [n_snapshots * n_components, n_nodes]
            Node-wise selection matrix.
        """
        component_blocks = []

        if dof_layout == "interleaved":
            for c in range(n_components):
                component_blocks.append(X_full[:, c::n_components])
        else:  # blocked
            for c in range(n_components):
                start = c * n_nodes
                stop = (c + 1) * n_nodes
                component_blocks.append(X_full[:, start:stop])

        X_nodes = np.vstack(component_blocks)

        if X_nodes.shape[1] != n_nodes:
            raise RuntimeError(
                f"Internal error while building node selection matrix. "
                f"Expected {n_nodes} columns, got {X_nodes.shape[1]}."
            )
        return X_nodes

    def _component_dof_index(self, node_idx: int, comp_idx: int) -> int:
        """Map (node, component) -> full-state DOF index."""
        if self.sensor_mode == "scalar":
            return int(node_idx)

        if self.dof_layout == "interleaved":
            return self.n_components * node_idx + comp_idx

        # blocked
        return comp_idx * self.n_nodes + node_idx

    def _preferred_sensor_count(self) -> int:
        """Preferred number of sensors from cumulative singular-value energy."""
        if self.s is None:
            raise RuntimeError("Call perform_svd() before requesting preferred sensor count.")

        s = np.sqrt(np.maximum(self.s, 0.0))
        s_mass = np.cumsum(s)
        if s_mass[-1] == 0.0:
            return min(self.n_candidate_locations, 1)

        idx = np.where(s_mass / s_mass[-1] > (1.0 - self.err_cap))[0]
        if len(idx) == 0:
            return min(self.n_candidate_locations, len(s))
        return int(idx[0] + 1)

    def perform_svd(self) -> None:
        """Compute the SVD of the Gram matrix on the selection snapshot matrix."""
        X = self.selection_snapshot_matrix
        self.u, self.s, self.v = LA.svd(X.T @ X, full_matrices=True)

    def plot_singular(self, err_cap: float = 0.00002) -> None:
        """Plot singular-value energy and cumulative fraction."""
        if self.s is None:
            self.perform_svd()

        self.err_cap = err_cap

        s = np.sqrt(np.maximum(self.s, 0.0))
        s_mass = np.cumsum(s)

        if s_mass[-1] == 0.0:
            self.s_count = 1
        else:
            self.s_count = [i for i, ss in enumerate(s_mass) if ss / s_mass[-1] > (1 - self.err_cap)][0] + 1

        print(f"The preferred number of sensors is: {self.s_count}")

        plt.figure()
        plt.scatter(np.arange(len(s)), s_mass / np.sum(s) if np.sum(s) != 0 else s_mass)
        plt.show()

        plt.figure()
        residual = 1 - s_mass / np.sum(s) if np.sum(s) != 0 else np.zeros_like(s_mass)
        plt.semilogy(np.arange(len(s)), residual, "o-")
        plt.show()

    def _fit_ps_model(self, n_sensors: int) -> None:
        """Fit SSPOR on the selection snapshot matrix."""
        self.ps_model = ps.SSPOR(
            ps.basis.SVD(self.n_SVD_basis, random_state=self.seed),
            n_sensors=n_sensors,
        )
        X = self._safe_normalize(self.selection_snapshot_matrix)
        self.ps_model.fit(X, seed=self.seed)

    def _fill_gaps_scalar(self, initial_indices: NDArray, target_count: int) -> NDArray:
        """Original index-midpoint gap filling for scalar channels."""
        sp = np.sort(np.asarray(initial_indices, dtype=int))
        current_count = len(sp)

        extended_subset = np.concatenate(([0], sp, [self.nx]))

        while current_count < target_count:
            gaps = np.diff(extended_subset)
            max_gap_index = int(np.argmax(gaps))
            point1 = int(extended_subset[max_gap_index])
            point2 = int(extended_subset[max_gap_index + 1])
            midpoint = (point1 + point2) // 2

            if midpoint in sp:
                # Fallback if integer midpoint already exists
                candidates = np.setdiff1d(np.arange(self.nx), sp)
                if len(candidates) == 0:
                    break
                midpoint = int(candidates[np.argmax(np.min(np.abs(candidates[:, None] - sp[None, :]), axis=1))])

            sp = np.sort(np.append(sp, midpoint))
            extended_subset = np.sort(np.append(extended_subset, midpoint))
            current_count += 1

        return sp

    def _fill_gaps_vector_nodes(self, initial_nodes: NDArray, target_count: int) -> NDArray:
        """Gap filling for vector-node mode.

        If node_coords is available:
            insert the unsensed node farthest from the current sensor set.
        Otherwise:
            fallback to midpoint filling in node-index space.
        """
        nodes = np.sort(np.asarray(initial_nodes, dtype=int))

        if self.node_coords is None:
            warnings.warn(
                "Vector-node mode gap filling is using node-index spacing because "
                "node_coords was not provided. For a real mesh, provide data['node_coords'] "
                "for geometry-aware spacing.",
                RuntimeWarning,
            )

            current_count = len(nodes)
            extended_subset = np.concatenate(([0], nodes, [self.n_nodes]))

            while current_count < target_count:
                gaps = np.diff(extended_subset)
                max_gap_index = int(np.argmax(gaps))
                point1 = int(extended_subset[max_gap_index])
                point2 = int(extended_subset[max_gap_index + 1])
                midpoint = (point1 + point2) // 2

                if midpoint in nodes:
                    candidates = np.setdiff1d(np.arange(self.n_nodes), nodes)
                    if len(candidates) == 0:
                        break
                    midpoint = int(candidates[np.argmax(np.min(np.abs(candidates[:, None] - nodes[None, :]), axis=1))])

                nodes = np.sort(np.append(nodes, midpoint))
                extended_subset = np.sort(np.append(extended_subset, midpoint))
                current_count += 1

            return nodes

        # Geometry-aware farthest-point insertion
        coords = np.asarray(self.node_coords, dtype=float)
        while len(nodes) < target_count:
            candidates = np.setdiff1d(np.arange(self.n_nodes), nodes)
            if len(candidates) == 0:
                break

            sensed_coords = coords[nodes]  # [k, dim]
            cand_coords = coords[candidates]  # [m, dim]

            # Distance from each candidate to nearest sensed node
            dists = np.linalg.norm(cand_coords[:, None, :] - sensed_coords[None, :, :], axis=2)
            nearest = np.min(dists, axis=1)
            next_node = int(candidates[np.argmax(nearest)])

            nodes = np.sort(np.append(nodes, next_node))

        return nodes

    def _build_C_matrix(self) -> None:
        """Build the measurement matrix C."""
        sp = np.asarray(self.sensor_placement, dtype=int)

        if self.sensor_mode == "scalar":
            C_Mat = np.zeros((len(sp), self.nx), dtype=float)
            C_Mat[np.arange(len(sp)), sp] = 1.0
            self.C_Mat = C_Mat
            self.sensor_dof_indices = sp.copy()
            return

        # vector_nodes mode
        dof = self.full_snapshot_matrix.shape[1]
        n_rows = len(sp) * self.n_components
        C_Mat = np.zeros((n_rows, dof), dtype=float)

        dof_indices = []
        row = 0
        for node in sp:
            for c in range(self.n_components):
                j = self._component_dof_index(int(node), c)
                C_Mat[row, j] = 1.0
                dof_indices.append(j)
                row += 1

        self.C_Mat = C_Mat
        self.sensor_dof_indices = np.asarray(dof_indices, dtype=int)

    def _build_full_state_A_matrix(self) -> None:
        """Build a full-state reconstruction basis from current snapshots."""
        X = self._safe_normalize(self.full_snapshot_matrix)
        _, _, Vh = LA.svd(X, full_matrices=False)
        self.A_Mat = Vh.T[:, : self.n_A_basis]

    def _build_A_matrix(self) -> None:
        """Build the reconstruction basis A.

        Scalar mode:
            preserve the original behavior and use the SSPOR/SVD basis on channels.

        Vector-node mode:
            compute a full-state basis on the actual DOF snapshots so that C @ A
            is well-defined with C built on full DOFs.
        """
        if self.sensor_mode == "scalar":
            if self.ps_model is None or not hasattr(self.ps_model, "basis_matrix_"):
                raise RuntimeError("SSPOR model is not fit. Call opt_sensor_loc() first.")
            self.A_Mat = self.ps_model.basis_matrix_[:, : self.n_A_basis]
            return

        self._build_full_state_A_matrix()

    def opt_sensor_loc(self, num_sensors: int | None = None, fill_gaps: bool = True) -> None:
        """Compute optimal sensor locations.

        Parameters
        ----------
        num_sensors : int or None
            Target number of sensors.
            - scalar mode: number of scalar channels
            - vector_nodes mode: number of physical nodes
        fill_gaps : bool
            If True and ``num_sensors > s_count``, fill the largest gaps.
            In vector-node mode, geometry-aware filling is used if node_coords is provided.
        """
        if self.s is None:
            self.perform_svd()

        self.s_count = self._preferred_sensor_count()

        if num_sensors is None and not fill_gaps:
            initial_count = self.s_count
            target_count = self.s_count

        elif num_sensors is not None and not fill_gaps:
            initial_count = int(num_sensors)
            target_count = int(num_sensors)

        elif num_sensors is None and fill_gaps:
            raise ValueError("Cannot have gap filling with no preferred sensor count")

        else:
            initial_count = self.s_count
            target_count = int(num_sensors)

        if initial_count <= 0:
            raise ValueError("Number of sensors must be positive.")

        self._fit_ps_model(n_sensors=initial_count)
        selected = np.sort(self.ps_model.get_selected_sensors()[:initial_count])

        if target_count > initial_count:
            if self.sensor_mode == "scalar":
                selected = self._fill_gaps_scalar(selected, target_count)
            else:
                selected = self._fill_gaps_vector_nodes(selected, target_count)

        self.sensor_placement = np.asarray(selected, dtype=int)

        if self.sensor_mode == "scalar":
            print(f"Sensors are in {self.sensor_placement}")
        else:
            print(f"Sensor nodes are {self.sensor_placement}")

        self._build_C_matrix()
        self._build_A_matrix()

        Theta = self.C_Mat @ self.A_Mat
        self.pinv_Theta = np.linalg.pinv(Theta)

    def set_sensor_placement(
        self,
        sensor_placement: NDArray,
        sensor_dof_indices: NDArray | None = None,
    ) -> None:
        """Reuse an existing sensor placement and fit a basis on current data.

        This is intended for cluster-local basis construction after a global
        sensor layout has already been selected.  It does not refit SSPOR or
        change the selected sensor locations.
        """
        self.sensor_placement = np.asarray(sensor_placement, dtype=int)

        if self.sensor_mode == "scalar":
            if np.any((self.sensor_placement < 0) | (self.sensor_placement >= self.nx)):
                raise ValueError("Scalar sensor placement contains out-of-range indices.")
        else:
            if np.any((self.sensor_placement < 0) | (self.sensor_placement >= self.n_nodes)):
                raise ValueError("Vector-node sensor placement contains out-of-range node indices.")

        if sensor_dof_indices is None:
            self._build_C_matrix()
        else:
            dof_indices = np.asarray(sensor_dof_indices, dtype=int)
            dof = self.full_snapshot_matrix.shape[1]
            if np.any((dof_indices < 0) | (dof_indices >= dof)):
                raise ValueError("Sensor DOF indices contain out-of-range values.")

            expected_rows = len(self.sensor_placement)
            if self.sensor_mode == "vector_nodes":
                expected_rows *= self.n_components

            if len(dof_indices) != expected_rows:
                raise ValueError(
                    f"Expected {expected_rows} sensor DOF indices, got {len(dof_indices)}."
                )

            C_Mat = np.zeros((len(dof_indices), dof), dtype=float)
            C_Mat[np.arange(len(dof_indices)), dof_indices] = 1.0
            self.C_Mat = C_Mat
            self.sensor_dof_indices = dof_indices

        self._build_full_state_A_matrix()

        Theta = self.C_Mat @ self.A_Mat
        self.pinv_Theta = np.linalg.pinv(Theta)

    def _slice_sparse(self, data_arr: NDArray) -> NDArray:
        """Extract sparse measurements from a single array."""
        data_arr = np.asarray(data_arr)

        if self.sensor_placement is None:
            raise RuntimeError("Call opt_sensor_loc() first.")

        if self.sensor_mode == "scalar":
            return data_arr[..., self.sensor_placement]

        return data_arr[..., self.sensor_dof_indices]

    def apply_sensors(
        self,
        data_train: NDArray,
        data_val: NDArray,
        data_test: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Extract sparse measurements at sensor locations.

        Returns
        -------
        Scalar mode:
            Each returned array has the same leading dimensions as input,
            with last axis = number of selected scalar channels.

        Vector-node mode:
            Each returned array has the same leading dimensions as input,
            with last axis = number_of_selected_nodes * n_components.
            The last axis is grouped by node:
                [node0_comp0, node0_comp1, ..., node1_comp0, node1_comp1, ...]
            For 2D interleaved displacement:
                [ux(node0), uy(node0), ux(node1), uy(node1), ...]
        """
        RS_train = self._slice_sparse(data_train)
        RS_val = self._slice_sparse(data_val)
        RS_test = self._slice_sparse(data_test)
        return RS_train, RS_val, RS_test

    @staticmethod
    def _resolve_matrix_path(path: str, matrix_subdir: str) -> str:
        if os.path.isabs(matrix_subdir):
            return os.path.abspath(matrix_subdir)
        return os.path.abspath(os.path.join(path, matrix_subdir))

    def load(
        self,
        path: str,
        matrix_subdir: str = "compression_matrices",
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Load pre-computed compression matrices from disk.

        Backward compatible with the original saved files:
            A_Mat.npy
            C_Mat.npy
            sensor_placement.npy

        If present, also loads:
            sensor_dof_indices.npy
            sensor_mode.npy
            n_components.npy
        """
        path = self._resolve_matrix_path(path, matrix_subdir)

        self.A_Mat = np.load(os.path.join(path, "A_Mat.npy"))
        self.C_Mat = np.load(os.path.join(path, "C_Mat.npy"))
        self.sensor_placement = np.load(os.path.join(path, "sensor_placement.npy"))

        sensor_dof_path = os.path.join(path, "sensor_dof_indices.npy")
        sensor_mode_path = os.path.join(path, "sensor_mode.npy")
        n_components_path = os.path.join(path, "n_components.npy")

        if os.path.exists(sensor_mode_path):
            self.sensor_mode = str(np.load(sensor_mode_path, allow_pickle=True).item())

        if os.path.exists(n_components_path):
            self.n_components = int(np.load(n_components_path))

        if os.path.exists(sensor_dof_path):
            self.sensor_dof_indices = np.load(sensor_dof_path)
        else:
            # Old scalar checkpoints
            self.sensor_dof_indices = self.sensor_placement.copy()

        Theta = self.C_Mat @ self.A_Mat
        self.pinv_Theta = np.linalg.pinv(Theta)

        return self.A_Mat, self.C_Mat, self.pinv_Theta, self.sensor_placement

    def save(self, path: str, matrix_subdir: str = "compression_matrices") -> None:
        """Save compression matrices to disk.

        Preserves the original files and adds a few optional metadata files.
        """
        path = self._resolve_matrix_path(path, matrix_subdir)
        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, "A_Mat.npy"), self.A_Mat)
        np.save(os.path.join(path, "C_Mat.npy"), self.C_Mat)
        np.save(os.path.join(path, "sensor_placement.npy"), self.sensor_placement)

        if self.sensor_dof_indices is not None:
            np.save(os.path.join(path, "sensor_dof_indices.npy"), self.sensor_dof_indices)

        np.save(os.path.join(path, "sensor_mode.npy"), np.array(self.sensor_mode, dtype=object))
        np.save(os.path.join(path, "n_components.npy"), np.array(self.n_components, dtype=int))
