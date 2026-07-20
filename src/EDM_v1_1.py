"""EDM_v1_1.py - EDM base + a CST mesh-gradient loss for 2-D full-state outputs.

Adds to the base ``EDM`` class the spatial regulariser used by 2-D examples:
for ``sensor_mode='vector_nodes'`` the 1-D second-difference spatial loss is
replaced by a constant-strain-triangle (CST) element-gradient loss assembled
from the shared mesh metadata.  The ``full_space`` module is only used to
lift predictions into full-DOF space for that gradient loss; primary losses
are computed in sparse measurement space (see ``EDM_v1_2.build_model``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from EDM import EDM as _EDMBase


def _resolve_mesh_json_path(config: dict[str, Any], mesh_json_path: str | None = None) -> str | None:
    """Resolve the CST mesh metadata path used by the 2-D gradient loss."""
    if mesh_json_path is not None:
        path = os.path.abspath(mesh_json_path)
        return path if os.path.exists(path) else None

    config_path = config.get("data", {}).get("mesh_json_path", None)
    if config_path is not None:
        path = os.path.abspath(config_path)
        return path if os.path.exists(path) else None

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / "beam_ellipse_shared.json",
        Path.cwd() / "Step_00_Create_Data" / "beam_ellipse_shared.json",
        Path.cwd().parent / "Step_00_Create_Data" / "beam_ellipse_shared.json",
        repo_root / "Examples" / "elliptic_hole_omega" / "Step_00_Create_Data" / "beam_ellipse_shared.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


class CSTElementGradient(nn.Module):
    """Map nodal DOFs to CST element gradients for displacement and velocity.

    Input
    -----
    ``X`` has shape ``[batch, time, 2 * ndof_field]`` where the first half is
    displacement DOFs and the second half is velocity DOFs.

    Output
    ------
    ``grad_X`` has shape ``[batch, time, n_elem * n_components * 2 * 2]``.
    For each element we store:
        - displacement gradients: [du_x/dx, du_x/dy, du_y/dx, du_y/dy]
        - velocity gradients:     [dv_x/dx, dv_x/dy, dv_y/dx, dv_y/dy]

    The gradients are multiplied by ``sqrt(area)`` so the downstream squared
    error behaves like an area-weighted integral over the mesh.
    """

    def __init__(
        self,
        mesh_json_path: str,
        n_components: int = 2,
        dof_layout: str = "interleaved",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if n_components != 2:
            raise ValueError(
                f"CSTElementGradient currently expects 2 components for the 2-D case; got {n_components}."
            )
        if dof_layout not in {"interleaved", "blocked"}:
            raise ValueError(f"Unsupported dof_layout={dof_layout!r}.")

        with open(mesh_json_path, "r") as f:
            mesh = json.load(f)

        coords = np.asarray(mesh["coords"], dtype=float)[:, :2]
        node_labels = np.asarray(mesh["node_labels"], dtype=int)
        tri_labels = np.asarray(mesh["tri3_connectivity_labels"], dtype=int)
        label_to_index = {int(lbl): i for i, lbl in enumerate(node_labels)}
        tri_conn = np.array(
            [[label_to_index[int(n)] for n in elem] for elem in tri_labels],
            dtype=np.int64,
        )

        grads = np.zeros((tri_conn.shape[0], 3, 2), dtype=np.float32)
        sqrt_area = np.zeros((tri_conn.shape[0], 1, 1), dtype=np.float32)

        for e, conn in enumerate(tri_conn):
            xy = coords[conn]
            x1, y1 = xy[0]
            x2, y2 = xy[1]
            x3, y3 = xy[2]

            twice_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if np.isclose(twice_area, 0.0):
                raise ValueError(f"Degenerate triangle detected at element {e}.")

            grads[e, :, 0] = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float32) / twice_area
            grads[e, :, 1] = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float32) / twice_area
            sqrt_area[e, 0, 0] = np.sqrt(abs(twice_area) / 2.0)

        self.n_nodes = coords.shape[0]
        self.n_components = n_components
        self.ndof_field = self.n_nodes * self.n_components
        self.dof_layout = dof_layout

        self.register_buffer("tri_conn", torch.tensor(tri_conn, dtype=torch.long, device=device))
        self.register_buffer("grad_shape", torch.tensor(grads, dtype=torch.float32, device=device))
        self.register_buffer("sqrt_area", torch.tensor(sqrt_area, dtype=torch.float32, device=device))

    @property
    def output_dim(self) -> int:
        """Total output dimension per time-step (for loss-weight scaling)."""
        n_elem = self.tri_conn.shape[0]
        # 2 fields (disp, vel) × n_components × 2 gradient directions × n_elem
        return n_elem * self.n_components * 2 * 2

    def _reshape_field(self, field_flat: torch.Tensor) -> torch.Tensor:
        batch, time, _ = field_flat.shape
        if self.dof_layout == "interleaved":
            return field_flat.view(batch, time, self.n_nodes, self.n_components)

        field = field_flat.view(batch, time, self.n_components, self.n_nodes)
        return field.permute(0, 1, 3, 2).contiguous()

    def _field_gradient(self, field_flat: torch.Tensor) -> torch.Tensor:
        field = self._reshape_field(field_flat)
        elem_vals = field[:, :, self.tri_conn, :]
        grads = torch.einsum("btenc,end->btecd", elem_vals, self.grad_shape)
        grads = grads * self.sqrt_area[None, None, :, :, :]
        return grads.reshape(field.shape[0], field.shape[1], -1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        disp = X[:, :, : self.ndof_field]
        vel = X[:, :, self.ndof_field :]

        grad_disp = self._field_gradient(disp)
        grad_vel = self._field_gradient(vel)
        return torch.cat([grad_disp, grad_vel], dim=-1)


class EDM(_EDMBase):
    """EDM with the original latent dynamics and a 2-D CST gradient loss.

    Changes from the base ``EDM``:

    *   Primary losses are computed in **sparse** measurement space.
        The ``full_space`` module is used only to lift into full-DOF space
        for the CST gradient regulariser — this prevents the reconstruction
        error from corrupting the training targets.

    *   The CST gradient loss weight is auto-scaled by the dimension ratio
        ``sparse_dim / gradient_output_dim`` so that a single
        ``Qs["SPATIALDIFF"]`` value is a reasonable starting point
        regardless of mesh resolution.
    """

    def __init__(
        self,
        A_mat: NDArray,
        pinv_Theta: NDArray,
        dt: float,
        t_max: float,
        config: dict[str, Any],
        device: str = "cpu",
        mesh_json_path: str | None = None,
    ) -> None:
        super().__init__(A_mat, pinv_Theta, dt, t_max, config, device)

        self.sensor_mode = config.get("sensors", {}).get("sensor_mode", "scalar")
        self.n_components = int(config.get("sensors", {}).get("n_components", 2))
        self.dof_layout = config.get("sensors", {}).get("dof_layout", "interleaved")

        self.mesh_json_path = _resolve_mesh_json_path(config, mesh_json_path)
        self.gradient_module: CSTElementGradient | None = None

        if self.sensor_mode == "vector_nodes":
            if self.mesh_json_path is None:
                raise FileNotFoundError(
                    "Could not resolve beam_ellipse_shared.json for the 2-D CST gradient loss. "
                    "Pass mesh_json_path explicitly or set data.mesh_json_path in the config."
                )
            self.gradient_module = CSTElementGradient(
                self.mesh_json_path,
                n_components=self.n_components,
                dof_layout=self.dof_layout,
                device=device,
            ).to(device)
