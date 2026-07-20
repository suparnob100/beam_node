"""Fourier-series beam vibration solver.

Solves the forced vibration of an Euler-Bernoulli beam using a truncated
Fourier-series representation.  All harmonic and spatial-mode loops are
fully vectorized with NumPy broadcasting / BLAS matrix multiplies.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from upsampler import fourier_upsample_add


class beam_problem:
    """Parametric beam vibration simulator.

    Parameters
    ----------
    nx : int
        Number of spatial grid points.
    nt : int
        Number of time steps per cycle.
    c_v, c_m : float
        Viscous and mass-proportional damping coefficients.
    ep : float
        Spatial width (epsilon) of the applied load.
    i_range : range
        Range of spatial mode indices.
    k_range : range
        Range of Fourier harmonic indices.
    upsample : object or None
        If not None, Fourier upsampling is applied to the output.
    t : NDArray or None
        Custom time vector.  If None, a uniform grid is created.
    max_dt : float or None
        Maximum time-step used when *upsample* is active.
    vars : list[str] or None
        Variable names (default ``["tau", "s"]``).
    """

    def __init__(
        self,
        nx: int = 101,
        nt: int = 200,
        c_v: float = 1.0,
        c_m: float = 0.001,
        ep: float = 0.02,
        i_range: range = range(1, 250),
        k_range: range = range(0, 90),
        upsample: object | None = None,
        t: NDArray | None = None,
        max_dt: float | None = None,
        vars: list[str] | None = None,
    ):
        self.nx = nx
        self.nt = nt
        self.c_v = c_v
        self.c_m = c_m
        self.ep = ep
        self.i_range = i_range
        self.k_range = k_range
        self.upsample = upsample
        self.t = t
        self.max_dt = max_dt
        self.vars = vars if vars is not None else ["tau", "s"]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def phi_k(x: NDArray, k: int | NDArray) -> NDArray:
        """Mode shape (simply-supported beam)."""
        return np.sqrt(2) * np.sin(k * np.pi * x)

    # ------------------------------------------------------------------
    # vectorized Fourier coefficient computation
    # ------------------------------------------------------------------
    def a_i(
        self,
        i: int,
        s: float,
        ep: float,
        tau: float,
        T: float,
        c_v: float,
        c_m: float,
        t: NDArray,
        omega: float,
    ) -> tuple[NDArray, NDArray]:
        """Compute Fourier coefficients for mode *i* (fully vectorized).

        Returns
        -------
        ai : NDArray
            Modal displacement time history.
        ai_dot : NDArray
            Modal velocity time history.
        """
        omega_i = (np.pi * i) ** 2

        # --- scalar helpers (mode-level, computed once) ----------------
        ci = c_v + c_m * omega_i ** 2

        if i * ep == 2:
            fi = (1.0 / np.sqrt(2)) * np.sin(i * s * np.pi)
        else:
            fi = (
                8.0 * np.sqrt(2) * np.sin(i * s * np.pi) * np.sin(i * np.pi * ep / 2)
            ) / (4.0 * np.pi * i * ep - i ** 3 * ep ** 3 * np.pi)

        # --- vectorised over all harmonics k in k_range ---------------
        if self.vars == ["omega"]:
            k_vals = np.arange(self.k_range.start, self.k_range.stop, dtype=float) + 1
        else:
            k_vals = np.arange(self.k_range.start, self.k_range.stop, dtype=float) + 1

        # dk array  (shape [K])
        resonant = np.isclose(T, k_vals * tau)
        dk_arr = np.where(
            resonant,
            (-1.0) ** k_vals / T,
            (2.0 * T ** 3 * np.cos(np.pi * k_vals) * np.sin(np.pi * k_vals * tau / T))
            / (T * (np.pi * k_vals * tau * T ** 2 - np.pi * k_vals ** 3 * tau ** 3)),
        )

        # alphak array  (shape [K])
        alphak_arr = omega_i ** 2 - (k_vals * omega) ** 2

        # CDi_k arrays  (shape [K])
        denom = (k_vals * ci * omega) ** 2 + alphak_arr ** 2
        Ci_k_arr = (dk_arr * alphak_arr * fi) / denom
        Di_k_arr = (k_vals * dk_arr * omega * ci * fi) / denom

        # --- sum over harmonics using broadcasting ---------------------
        if self.vars == ["omega"]:
            # k_vals already correct for omega mode
            kwt = k_vals[:, None] * omega * t[None, :]  # [K, n_t]
            sum_d = np.sum(
                Ci_k_arr[:, None] * np.cos(kwt) + Di_k_arr[:, None] * np.sin(kwt),
                axis=0,
            )
            sum_v = np.sum(
                (-k_vals[:, None] * omega) * Ci_k_arr[:, None] * np.sin(kwt)
                + (k_vals[:, None] * omega) * Di_k_arr[:, None] * np.cos(kwt),
                axis=0,
            )
            return (fi / (omega_i ** 2 * T)) * 0 + sum_d, sum_v
        else:
            kp1 = k_vals  # already shifted by +1 above
            kwt = kp1[:, None] * omega * t[None, :]  # [K, n_t]
            sum_d = np.sum(
                Ci_k_arr[:, None] * np.cos(kwt) + Di_k_arr[:, None] * np.sin(kwt),
                axis=0,
            )
            sum_v = np.sum(
                (-kp1[:, None] * omega) * Ci_k_arr[:, None] * np.sin(kwt)
                + (kp1[:, None] * omega) * Di_k_arr[:, None] * np.cos(kwt),
                axis=0,
            )
            return (fi / (omega_i ** 2 * T)) + sum_d, sum_v

    # ------------------------------------------------------------------
    # forcing function
    # ------------------------------------------------------------------
    def forcing_fn(
        self, tau: float, s: float, omega: float, t: NDArray
    ) -> NDArray:
        """Build the forcing vector over time."""

        def forcing_fn_t(t: NDArray, tau: float, T: float, K: int = 90) -> NDArray:
            k_vals = np.arange(1, K + 1, dtype=float)
            resonant = np.isclose(T, k_vals * tau)
            dk_arr = np.where(
                resonant,
                (-1.0) ** k_vals / T,
                (2.0 * T ** 3 * np.cos(np.pi * k_vals) * np.sin(np.pi * k_vals * tau / T))
                / (T * (np.pi * k_vals * tau * T ** 2 - np.pi * tau ** 3 * k_vals ** 3)),
            )
            # Broadcasting: dk [K] x cos [K, n_t] → sum → [n_t]
            ft = np.sum(dk_arr[:, None] * np.cos(2 * np.pi * k_vals[:, None] * t[None, :] / T), axis=0)
            ft += 1.0 / T
            return ft

        var_dict = {"tau": tau, "s": s, "omega": omega}
        n_dim = len(self.vars)
        ft = np.zeros([self.nt, n_dim + 1])
        if "omega" in self.vars:
            t_lin = np.linspace(0, 2 * np.pi / omega, t.shape[0])
            ft[:, 0] = forcing_fn_t(t_lin, tau, 2 * np.pi / omega).reshape(self.nt)
        else:
            ft[:, 0] = forcing_fn_t(t, tau, 2 * np.pi / omega).reshape(self.nt)
        for idx, var in enumerate(self.vars):
            ft[:, idx + 1] = var_dict[var]
        return ft

    # ------------------------------------------------------------------
    # main solver  (vectorized over spatial modes)
    # ------------------------------------------------------------------
    def solve(
        self, tau: float, s: float, omega: float, cycles: int = 1
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Solve beam vibration problem.

        Returns
        -------
        W_T : NDArray  [n_t, nx]
            Displacement field (transposed).
        Wdot_T : NDArray  [n_t, nx]
            Velocity field (transposed).
        ft : NDArray  [n_t, n_dim+1]
            Forcing vector.
        """
        T = 2 * np.pi / omega
        if self.t is not None:
            t = self.t
        else:
            t = np.linspace(0, T * cycles, self.nt * cycles + 1)

        x = np.linspace(0, 1, self.nx)

        # Pre-compute all mode shapes  [n_modes, nx]
        i_arr = np.array(list(self.i_range))
        phi_all = np.sqrt(2) * np.sin(i_arr[:, None] * np.pi * x[None, :])

        # Compute all modal coefficients  [n_modes, n_t]
        ai_all = np.empty((len(i_arr), t.shape[0]))
        ai_dot_all = np.empty((len(i_arr), t.shape[0]))
        for idx, i in enumerate(self.i_range):
            ai_all[idx], ai_dot_all[idx] = self.a_i(
                i, s, self.ep, tau, T, self.c_v, self.c_m, t, omega
            )

        # Single matrix multiply: phi_all.T [nx, n_modes] @ ai_all [n_modes, n_t]
        W = phi_all.T @ ai_all           # [nx, n_t]
        W_dot = phi_all.T @ ai_dot_all   # [nx, n_t]

        ft = self.forcing_fn(tau, s, omega, t[:-1])

        if self.upsample is not None:
            W = W[:, : int(T / self.max_dt)]
            W_dot = W_dot[:, : int(T / self.max_dt)]
            W_ret = np.zeros([self.nx, t.shape[0]])
            W_dot_ret = np.zeros([self.nx, t.shape[0]])

            for i in range(self.nx):
                W_ret[i] = fourier_upsample_add(W[i], t.shape[0] - W.shape[1])
                W_dot_ret[i] = fourier_upsample_add(W_dot[i], t.shape[0] - W.shape[1])

            return W_ret[:, :-1].T, W_dot_ret[:, :-1].T, ft

        else:
            return W[:, :-1].T, W_dot[:, :-1].T, ft
