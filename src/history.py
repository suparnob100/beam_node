"""history.py — causal forcing-history features for gated NODEs.

The instantaneous forcing value u(t) is ambiguous when the waveform shape
is unknown (the same force level occurs on the way up, on the way down,
between pulses, ...).  Instead of assuming any shape (harmonic, pulse,
...), we augment the latent ODE state with a small bank of *leaky
integrators* driven by u:

    h_dot_i = ( W_u u + b - h )_i / tau_i          (per-feature tau_i)

Each h_i is a causal low-pass summary of the forcing history at its own
learnable timescale tau_i.  The bank is initialised with log-spaced
timescales spanning [tau_min, t_max], so at epoch 0 the features already
cover fast transients through slow trends.  Everything is smooth in t,
so RK4 integrates it without special handling — the history features are
just extra latent dimensions with a *structured* (physics-agnostic but
interpretable) vector field.

No assumption about the forcing waveform is made anywhere in this file.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn


class LeakyHistoryBank(nn.Module):
    """Bank of ``n_hist`` leaky integrators driven by the control input.

    Parameters
    ----------
    n_control : int
        Width of the forcing input u.
    n_hist : int
        Number of history features h.
    dt : float
        Integration step of the outer RK4 (used only to pick sensible
        timescale bounds).
    t_max : float
        Trajectory horizon (upper bound for the slowest timescale init).
    tau_min_mult : float
        Fastest allowed timescale = ``tau_min_mult * dt``.  Keep >= 2 so
        the history ODE stays well-resolved by the outer integrator.
    """

    def __init__(
        self,
        n_control: int,
        n_hist: int,
        dt: float,
        t_max: float,
        tau_min_mult: float = 2.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_control = n_control
        self.n_hist = n_hist
        self.tau_min = tau_min_mult * dt

        # Log-spaced initial timescales in [4*dt, t_max].
        lo = math.log(max(4.0 * dt, self.tau_min * 1.01) - self.tau_min)
        hi = math.log(max(t_max, 8.0 * dt) - self.tau_min)
        init = torch.linspace(lo, hi, n_hist)
        # log_tau parametrises tau = softplus(log_tau) + tau_min.
        # softplus(x) ~ x for x >> 0 and ~ e^x for x << 0; the inverse
        # below is exact: softplus(log(e^y - 1)) = y.
        self.log_tau = nn.Parameter(torch.log(torch.expm1(init.exp())))

        self.W_u = nn.Linear(n_control, n_hist, bias=True)
        nn.init.normal_(self.W_u.weight, std=1.0 / math.sqrt(n_control))
        nn.init.zeros_(self.W_u.bias)

        self.to(device)

    @property
    def tau(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_tau) + self.tau_min

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return h_dot given current history features and forcing."""
        target = self.W_u(u)
        return (target - h) / self.tau

    def rk4_decay(self, dt: float) -> torch.Tensor:
        """Per-feature decay of one outer RK4 step for constant control.

        For ``h_dot = (target - h) / tau``, classical RK4 gives
        ``h_next = a*h + (1-a)*target`` with the fourth-order stability
        polynomial evaluated at ``-dt/tau``.  Using this factor keeps the
        periodic initializer exactly consistent with the NODE integrator.
        """
        r = torch.as_tensor(dt, dtype=self.tau.dtype, device=self.tau.device) / self.tau
        return 1.0 - r + 0.5 * r.square() - (r ** 3) / 6.0 + (r ** 4) / 24.0

    def rk4_step(self, h: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the history state by one constant-control RK4 step."""
        target = self.W_u(u)
        a = self.rk4_decay(dt).to(device=target.device, dtype=target.dtype)
        return a * h + (1.0 - a) * target

    def periodic_initial_state(self, U_period: torch.Tensor, dt: float) -> torch.Tensor:
        """Return the unique periodic history state at the start of a cycle.

        Parameters
        ----------
        U_period : torch.Tensor
            Forcing ordered from the desired start phase, shape ``[B,T,nu]``
            or ``[T,nu]``.  The returned state is the history immediately
            before ``U_period[:, 0]`` is applied.
        dt : float
            Outer integrator step.

        The computation is analytic and differentiable with respect to the
        history-bank parameters.  It avoids an artificial from-rest transient
        and is substantially cheaper than repeatedly rolling several cycles.
        """
        if U_period.ndim == 2:
            U_period = U_period.unsqueeze(0)
        if U_period.ndim != 3 or U_period.shape[-1] != self.n_control:
            raise ValueError(
                f"U_period must have shape [B,T,{self.n_control}] or "
                f"[T,{self.n_control}], got {tuple(U_period.shape)}"
            )

        T = U_period.shape[1]
        if T < 1:
            raise ValueError("U_period must contain at least one time step")

        # The geometric powers are sensitive to bf16/fp16 quantisation (for
        # example, bf16 cannot represent every integer exponent above 256).
        # Compute the initializer in fp32 even under autocast, then cast back.
        device_type = U_period.device.type
        autocast_supported = device_type in {"cpu", "cuda"}
        context = (
            torch.autocast(device_type=device_type, enabled=False)
            if autocast_supported else nullcontext()
        )
        with context:
            U32 = U_period.float()
            target = torch.nn.functional.linear(
                U32, self.W_u.weight.float(),
                None if self.W_u.bias is None else self.W_u.bias.float(),
            )
            tau = self.tau.float()
            r = torch.as_tensor(dt, dtype=torch.float32, device=target.device) / tau
            a = (1.0 - r + 0.5 * r.square() - (r ** 3) / 6.0 + (r ** 4) / 24.0)
            a = a.clamp(min=0.0, max=1.0 - 1e-7)
            exponents = torch.arange(T - 1, -1, -1, device=target.device, dtype=torch.float32)
            weights = (1.0 - a).unsqueeze(0) * a.unsqueeze(0).pow(exponents.unsqueeze(1))
            cycle_input = torch.einsum("bth,th->bh", target, weights)
            denom = (1.0 - a.pow(T)).clamp_min(torch.finfo(torch.float32).eps)
            h0 = cycle_input / denom
        return h0.to(dtype=U_period.dtype)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        with torch.no_grad():
            taus = self.tau.detach().cpu().numpy()
        return f"n_hist={self.n_hist}, tau range=[{taus.min():.4g}, {taus.max():.4g}]"
