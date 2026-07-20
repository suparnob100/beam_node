"""scikit-fem solver on the shared Gmsh TRI3 mesh with a concentrated load
at the top-midpoint node.

This version keeps the same mesh workflow as the shared-mesh setup but removes
boundary traction loading. The external force is applied only at the top node
closest to x = L/2, y = H.

Default temporal loading is a bounded sinusoid to make code-to-code comparison
cleaner. A pulse-train option is also provided for compatibility.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from skfem import MeshTri, ElementTriP1, ElementVector, Basis, BilinearForm
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import dot


@BilinearForm
def vector_mass(u, v, w):
    return dot(u, v)


def _fourier_pulse_train(t: NDArray, tau: float, T: float, K: int = 90) -> NDArray:
    t = np.asarray(t, dtype=float)
    k_vals = np.arange(1, K + 1, dtype=float)

    resonant = np.isclose(T, k_vals * tau)
    dk_arr = np.where(
        resonant,
        (-1.0) ** k_vals / T,
        (2.0 * T**3 * np.cos(np.pi * k_vals) * np.sin(np.pi * k_vals * tau / T))
        / (T * (np.pi * k_vals * tau * T**2 - np.pi * tau**3 * k_vals**3)),
    )

    ft = np.sum(
        dk_arr[:, None] * np.cos(2.0 * np.pi * k_vals[:, None] * t[None, :] / T),
        axis=0,
    )
    ft += 1.0 / T
    return ft


def _harmonic_signal(t: NDArray, omega: float) -> NDArray:
    return np.sin(omega * np.asarray(t, dtype=float))


def _load_shared_mesh_data(mesh_json_path: str) -> dict:
    if not os.path.exists(mesh_json_path):
        raise FileNotFoundError(
            'Shared mesh metadata not found: %s\nGenerate it first with common_gmsh_beam_mesh.py.'
            % mesh_json_path
        )
    with open(mesh_json_path, 'r') as f:
        return json.load(f)


class BeamFEAPointLoad:
    def __init__(
        self,
        mesh_json_path: str = 'beam_ellipse_shared.json',
        E: float = 210e9,
        nu: float = 0.3,
        rho: float = 7800.0,
    ):
        self.mesh_json_path = mesh_json_path
        self.E = E
        self.nu = nu
        self.rho = rho

        lam_3d, mu_3d = lame_parameters(E, nu)
        self.mu_ps = mu_3d
        self.lam_ps = 2.0 * lam_3d * mu_3d / (lam_3d + 2.0 * mu_3d)

        self._data = None
        self._mesh = None
        self._ib = None
        self._K = None
        self._M = None
        self._f_vec = None
        self._dofs_left = None
        self._sample_dof_indices = None
        self._x_sample = None
        self._node_labels = None
        self._label_to_index = None
        self._top_mid_node_label = None
        self._top_mid_node_index = None
        self._top_mid_y_dof = None

    def _assemble(self):
        if self._mesh is not None:
            return

        data = _load_shared_mesh_data(self.mesh_json_path)
        coords = np.asarray(data['coords'], dtype=float)
        node_labels = np.asarray(data['node_labels'], dtype=int)
        tri_conn_labels = np.asarray(data['tri3_connectivity_labels'], dtype=int)
        label_to_index = {int(lbl): i for i, lbl in enumerate(node_labels)}
        tri_conn = np.array(
            [[label_to_index[int(n)] for n in elem] for elem in tri_conn_labels],
            dtype=int,
        ).T

        p = coords[:, :2].T
        mesh = MeshTri(p, tri_conn)
        L = float(data['L'])
        H = float(data['H'])
        mesh = mesh.with_boundaries({
            'left': lambda x: np.isclose(x[0], 0.0),
            'right': lambda x: np.isclose(x[0], L),
            'top': lambda x: np.isclose(x[1], H),
            'bottom': lambda x: np.isclose(x[1], 0.0),
        })

        elem = ElementVector(ElementTriP1())
        ib = Basis(mesh, elem)

        K = linear_elasticity(self.lam_ps, self.mu_ps).assemble(ib)
        M = self.rho * vector_mass.assemble(ib)

        ndof = K.shape[0]
        f_vec = np.zeros(ndof, dtype=float)
        dof_map = ib.nodal_dofs

        top_labels = np.asarray(data['boundary_node_labels']['top'], dtype=int)
        top_indices = np.array([label_to_index[int(lbl)] for lbl in top_labels], dtype=int)
        top_x = coords[top_indices, 0]
        i_mid = int(np.argmin(np.abs(top_x - 0.5 * L)))
        top_mid_node_label = int(top_labels[i_mid])
        top_mid_node_index = int(top_indices[i_mid])
        top_mid_y_dof = int(dof_map[1, top_mid_node_index])
        f_vec[top_mid_y_dof] = 1.0

        dofs_left = ib.get_dofs('left').all()

        sample_labels = np.asarray(data['top_sample_node_labels'], dtype=int)
        sample_indices = np.array([label_to_index[int(lbl)] for lbl in sample_labels], dtype=int)
        sample_dof_indices = dof_map[1, sample_indices]
        x_sample = np.asarray(data['top_sample_x'], dtype=float)

        self._data = data
        self._mesh = mesh
        self._ib = ib
        self._K = K
        self._M = M
        self._f_vec = f_vec
        self._dofs_left = dofs_left
        self._sample_dof_indices = sample_dof_indices
        self._x_sample = x_sample
        self._node_labels = node_labels
        self._label_to_index = label_to_index
        self._top_mid_node_label = top_mid_node_label
        self._top_mid_node_index = top_mid_node_index
        self._top_mid_y_dof = top_mid_y_dof

    @property
    def mesh(self):
        self._assemble()
        return self._mesh

    @property
    def top_mid_node_label(self) -> int:
        self._assemble()
        return self._top_mid_node_label

    @property
    def top_mid_y_dof(self) -> int:
        self._assemble()
        return self._top_mid_y_dof

    def solve(
        self,
        omega: float,
        tau: float | None = None,
        nt: int = 400,
        F0: float = 1e6,
        alpha_d: float = 0.5,
        beta_d: float = 1e-6,
        t: NDArray | None = None,
        n_periods: int = 20,
        n_fourier_terms: int = 90,
        return_last_period_only: bool = True,
        load_type: Literal['harmonic', 'pulse'] = 'harmonic',
    ):
        self._assemble()

        if omega <= 0.0:
            raise ValueError('omega must be positive.')

        K = self._K
        M = self._M
        f_vec = self._f_vec
        dofs_left = self._dofs_left
        sample_dof_indices = self._sample_dof_indices
        ndof = K.shape[0]

        T = 2.0 * np.pi / omega
        if load_type == 'pulse':
            if tau is None or tau <= 0.0:
                raise ValueError('tau must be positive for pulse loading.')
            if tau >= T:
                raise ValueError('Need tau < T. Got tau=%e, T=%e.' % (tau, T))

        if t is None:
            nt_per_period = int(nt)
            if nt_per_period < 2:
                raise ValueError('nt must be at least 2.')
            n_steps_total = n_periods * nt_per_period
            t = np.linspace(0.0, n_periods * T, n_steps_total + 1)
            internally_generated_t = True
        else:
            t = np.asarray(t, dtype=float).ravel()
            if t.ndim != 1 or len(t) < 2:
                raise ValueError('t must be a 1-D array with length >= 2.')
            n_steps_total = len(t) - 1
            nt_per_period = None
            internally_generated_t = False

        dt_arr = np.diff(t)
        if not np.allclose(dt_arr, dt_arr[0], rtol=1e-10, atol=1e-14):
            raise ValueError('This Newmark implementation requires a uniform time step.')
        dt = float(dt_arr[0])

        if load_type == 'harmonic':
            F_t = F0 * _harmonic_signal(t, omega)
        elif load_type == 'pulse':
            F_t = F0 * _fourier_pulse_train(t, tau, T, K=n_fourier_terms)
        else:
            raise ValueError("load_type must be 'harmonic' or 'pulse'.")

        C = alpha_d * M + beta_d * K

        beta_nm = 0.25
        gamma_nm = 0.5

        a0 = 1.0 / (beta_nm * dt**2)
        a1 = gamma_nm / (beta_nm * dt)
        a2 = 1.0 / (beta_nm * dt)
        a3 = 1.0 / (2.0 * beta_nm) - 1.0
        a4 = gamma_nm / beta_nm - 1.0
        a5 = dt * (gamma_nm / (2.0 * beta_nm) - 1.0)

        K_eff = K + a0 * M + a1 * C

        is_free = np.ones(ndof, dtype=bool)
        is_free[dofs_left] = False
        free = np.where(is_free)[0]

        K_ff = K[free][:, free].tocsc()
        M_ff = M[free][:, free].tocsc()
        C_ff = C[free][:, free].tocsc()
        K_eff_ff = K_eff[free][:, free].tocsc()
        f_free_base = f_vec[free]

        u = np.zeros(ndof)
        v = np.zeros(ndof)
        acc = np.zeros(ndof)

        F_ext0_free = F_t[0] * f_free_base
        acc[free] = spsolve(M_ff, F_ext0_free - C_ff @ v[free] - K_ff @ u[free])

        W_all = np.zeros((n_steps_total, len(sample_dof_indices)))
        Wdot_all = np.zeros((n_steps_total, len(sample_dof_indices)))
        W_full_all = np.zeros((n_steps_total, ndof))
        Wdot_full_all = np.zeros((n_steps_total, ndof))

        for step in range(n_steps_total):
            F_ext_free = F_t[step + 1] * f_free_base

            rhs = (
                F_ext_free
                + M_ff @ (a0 * u[free] + a2 * v[free] + a3 * acc[free])
                + C_ff @ (a1 * u[free] + a4 * v[free] + a5 * acc[free])
            )

            u_new_free = spsolve(K_eff_ff, rhs)
            acc_new_free = a0 * (u_new_free - u[free]) - a2 * v[free] - a3 * acc[free]
            v_new_free = v[free] + dt * ((1.0 - gamma_nm) * acc[free] + gamma_nm * acc_new_free)

            u[free] = u_new_free
            v[free] = v_new_free
            acc[free] = acc_new_free

            W_all[step] = u[sample_dof_indices]
            Wdot_all[step] = v[sample_dof_indices]
            W_full_all[step] = u.copy()
            Wdot_full_all[step] = v.copy()

        if internally_generated_t and return_last_period_only:
            start = (n_periods - 1) * nt_per_period
            stop = n_periods * nt_per_period

            W = W_all[start:stop]
            Wdot = Wdot_all[start:stop]
            W_full = W_full_all[start:stop]
            Wdot_full = Wdot_full_all[start:stop]
            t_out = t[start + 1:stop + 1] - t[start + 1]
            ft = np.column_stack([F_t[start + 1:stop + 1], np.full(nt_per_period, omega)])
        else:
            W = W_all
            Wdot = Wdot_all
            W_full = W_full_all
            Wdot_full = Wdot_full_all
            t_out = t[1:]
            ft = np.column_stack([F_t[1:n_steps_total + 1], np.full(n_steps_total, omega)])

        return W, Wdot, ft, self._mesh, self._ib, W_full, Wdot_full, t_out, t, W_full_all, Wdot_full_all


# if __name__ == '__main__':
#     solver = BeamFEAPointLoad(mesh_json_path='beam_ellipse_shared.json')
#     omega = 100.0
#     W, Wdot, ft, mesh, ib, W_full, Wdot_full, t = solver.solve(
#         omega=omega,
#         nt=400,
#         n_periods=20,
#         alpha_d=0.5,
#         beta_d=1e-6,
#         load_type='harmonic',
#     )
#     print('W shape:', W.shape)
#     print('Returned time shape:', t.shape)
#     print('Top-midpoint node label:', solver.top_mid_node_label)
#     print('Top-midpoint y-DOF:', solver.top_mid_y_dof)
