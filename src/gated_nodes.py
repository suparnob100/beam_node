"""gated_nodes.py — the rbf_moe forcing-history gated NODE (no wall-clock gate).

``RBFHistoryMoE_NODE`` exposes the neuromancer-compatible signature

    forward(x, u, t) -> x_dot        # t accepted but NEVER used

with ``in_features`` / ``out_features`` so ``integrators.RK4`` and
``CheckpointedRK4`` treat it exactly like a plain NODE.  The history
features are carried inside the latent state:

    x = [ z (lat_space) | h (n_hist) ]

so the encoder outputs only z; ``EDM_v1_2.EDM.build_model`` appends the
periodic history state h0 from the aligned forcing cycle (``U_period``).

RBFHistoryMoE_NODE ("rbf_moe"): K experts on [z, u], an RBFGate over the
causal forcing-history features [h, u_gate] (optionally with periodic phase
features of the gate-visible controls).  Uses a sigma floor, temperature
control, and can be paired with a physically non-uniform expert-usage prior.
``GateProbe`` recomputes the gate weights along a rollout for the usage
prior and diagnostics.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from history import LeakyHistoryBank
from gates import RBFGate


def _normalise_gate_control_indices(
    n_control: int,
    use_u_in_gate: bool,
    gate_control_indices: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    """Validate which instantaneous control channels are visible to the gate.

    By default only the first, time-varying forcing channel is exposed.  This
    prevents constant parameter channels (for example omega) from turning the
    gate into a trajectory-identity classifier.  Experts still receive the
    complete control vector.
    """
    if not use_u_in_gate:
        return ()
    if gate_control_indices is None:
        gate_control_indices = (0,)
    indices = tuple(int(i) for i in gate_control_indices)
    if not indices:
        return ()
    if len(set(indices)) != len(indices) or any(i < 0 or i >= n_control for i in indices):
        raise ValueError(
            f"gate_control_indices must contain unique values in [0,{n_control}), "
            f"got {indices}"
        )
    return indices


def _periodic_phase_features(
    t: torch.Tensor | None,
    reference: torch.Tensor,
    t_max: float,
    n_harmonics: int,
) -> torch.Tensor:
    """Return periodic sin/cos phase features with the same leading shape as reference."""
    if t is None:
        raise ValueError("The gate requires time t when use_phase_in_gate=True.")
    if n_harmonics < 1:
        raise ValueError("gate_phase_harmonics must be at least 1.")

    target_shape = reference.shape[:-1]
    phase = t
    if phase.ndim == reference.ndim and phase.shape[-1] == 1:
        phase = phase[..., 0]
    phase = torch.broadcast_to(phase, target_shape)
    phase = torch.remainder(phase, float(t_max)) / float(t_max)

    features = []
    for harmonic in range(1, n_harmonics + 1):
        angle = 2.0 * torch.pi * harmonic * phase
        features.extend((torch.sin(angle), torch.cos(angle)))
    return torch.stack(features, dim=-1)


def _expert_mlp(in_f: int, out_f: int, n_layers: int, n_units: int) -> nn.Sequential:
    """Plain SiLU MLP.  SiLU (not ReLU): RK4 wants a C^1 vector field.
    No dropout: RK4 evaluates the field 4x per step and per-stage random
    masks make the integrated field self-inconsistent."""
    sizes = [in_f] + [n_units] * n_layers + [out_f]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class _HistoryStateMixin:
    """Split/merge helpers for the augmented state [z | h]."""

    lat_space: int
    n_hist: int

    def split_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.lat_space], x[..., self.lat_space :]


# ─────────────────────────────────────────────────────────────────────────
#  rbf_moe — RBF gate over forcing-history features
# ─────────────────────────────────────────────────────────────────────────

class RBFHistoryMoE_NODE(nn.Module, _HistoryStateMixin):
    """K experts on [z, u], blended by an RBF gate on [h, u_gate].

    Uses a sigma floor and temperature control.  A weak, physically correct
    usage prior may be used, but a uniform 1/K balance target is inappropriate
    when the pulse occupies only a small fraction of the cycle.
    """

    method_name = "rbf_moe"
    augment_state = True

    def __init__(
        self,
        lat_space: int,
        n_control: int,
        n_layers: int,
        n_units: int,
        dt: float,
        t_max: float,
        n_experts: int = 3,
        n_hist: int = 8,
        sigma_min: float = 0.25,
        use_u_in_gate: bool = True,
        gate_control_indices: tuple[int, ...] | list[int] | None = None,
        use_phase_in_gate: bool = False,
        gate_phase_harmonics: int = 1,
        temperature_init: float = 2.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.lat_space = lat_space
        self.n_control = n_control
        self.n_hist = n_hist
        self.n_experts = n_experts
        self.use_u_in_gate = use_u_in_gate
        self.use_phase_in_gate = bool(use_phase_in_gate)
        self.gate_phase_harmonics = int(gate_phase_harmonics)
        self.gate_control_indices = _normalise_gate_control_indices(
            n_control, use_u_in_gate, gate_control_indices
        )
        self.dt = float(dt)
        self.t_max = float(t_max)

        self.state_dim = lat_space + n_hist
        self.in_features = self.state_dim + n_control
        self.out_features = self.state_dim

        self.history = LeakyHistoryBank(n_control, n_hist, dt, t_max, device=device)
        gate_in = (
            n_hist
            + len(self.gate_control_indices)
            + (2 * self.gate_phase_harmonics if self.use_phase_in_gate else 0)
        )
        self.gate = RBFGate(
            gate_in, n_experts, sigma_min=sigma_min,
            temperature_init=temperature_init, device=device,
        )
        self.experts = nn.ModuleList(
            [_expert_mlp(lat_space + n_control, lat_space, n_layers, n_units) for _ in range(n_experts)]
        )
        self.to(device)

    def gate_features(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [h]
        if self.gate_control_indices:
            features.append(u[..., list(self.gate_control_indices)])
        if self.use_phase_in_gate:
            features.append(
                _periodic_phase_features(
                    t, h, self.t_max, self.gate_phase_harmonics
                )
            )
        return torch.cat(features, dim=-1)

    def periodic_history_initial_state(self, U_period: torch.Tensor) -> torch.Tensor:
        """Periodic history state at the phase represented by ``U_period[:,0]``."""
        return self.history.periodic_initial_state(U_period, self.dt)

    def gate_weights(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.gate(self.gate_features(h, u, t))

    @torch.no_grad()
    def gate_feature_samples(
        self, U: torch.Tensor, max_samples: int = 20000, flatten: bool = True
    ) -> torch.Tensor:
        """Roll the (current) history bank over forcing ``U`` and return the
        [h, u] gate features the RBF gate will see, for data-driven seeding.

        ``U``: [N, T, n_control] training forcing.  Each trajectory is treated
        as one periodic cycle.  The history starts from its differentiable
        periodic fixed point, eliminating the artificial fill-up transient at
        t=0, and is advanced with the same RK4 discrete update as training.

        ``flatten=True`` returns a subsampled [M, gate_in] pool; ``False``
        keeps the trajectory structure, [N, T, gate_in] (no subsampling),
        for per-trajectory seeding in ``RBFGate.init_from_features``.
        """
        dev = next(self.parameters()).device
        U = U.detach().to(dev, torch.float32)
        if U.ndim == 2:
            U = U.unsqueeze(0)
        B, T, _ = U.shape
        h = self.periodic_history_initial_state(U)
        dt = self.dt
        feats = []
        for step in range(T):
            u_t = U[:, step, :]
            t_t = torch.full(
                (B, 1),
                fill_value=(step * self.t_max / T),
                dtype=U.dtype,
                device=U.device,
            )
            feats.append(self.gate_features(h, u_t, t_t))
            h = self.history.rk4_step(h, u_t, dt)
        if not flatten:
            return torch.stack(feats, dim=1)  # [B, T, gate_in]
        feats = torch.cat(feats, dim=0)  # [B*T, gate_in]
        if feats.shape[0] > max_samples:
            idx = torch.randperm(feats.shape[0], device=feats.device)[:max_samples]
            feats = feats[idx]
        return feats

    def forward(self, x: torch.Tensor, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z, h = self.split_state(x)
        w = self.gate_weights(h, u, t)
        zu = torch.cat([z, u], dim=-1)
        outs = torch.stack([f(zu) for f in self.experts], dim=-1)
        z_dot = torch.einsum("blk,bk->bl", outs, w)
        h_dot = self.history(h, u)
        return torch.cat([z_dot, h_dot], dim=-1)


# ─────────────────────────────────────────────────────────────────────────
#  Graph probe — recomputes gate weights along a rollout for the
#  importance loss and for monitoring.  Shares parameters with the NODE.
# ─────────────────────────────────────────────────────────────────────────

class GateProbe(nn.Module):
    """Node wrapper: (LS_trajectory, U_trajectory) -> per-step gate weights
    AND their broadcast batch/time mean.

    Outputs
    -------
    gate_w      [B, T, K]   per-step weights (for plotting / diagnostics)
    gate_w_mean [B, T, K]   every entry equals the batch-time mean usage
                            of that expert.  Penalising
                            ``(gate_w_mean == 1/K)^2`` in the objective is
                            exactly the importance load-balance loss (the
                            broadcast keeps neuromancer's shape checks
                            happy while gradients still flow to the gate).
    """

    def __init__(self, node: nn.Module, detach_features: bool = True) -> None:
        super().__init__()
        # Keep a non-registered reference: the NODE is already registered by
        # NODE_System. Registering it again here duplicates checkpoint paths.
        object.__setattr__(self, "node", node)
        self.detach_features = detach_features

    def forward(
        self,
        LS: torch.Tensor,
        U: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = U.shape[1]
        LS = LS[:, :T, :]  # rollout has nsteps+1 states; align to controls
        if getattr(self.node, "augment_state", False):
            feats_a = LS[..., self.node.lat_space :]     # h
        else:
            feats_a = LS                                  # z
        if self.detach_features:
            feats_a = feats_a.detach()
        w = self.node.gate(self.node.gate_features(feats_a, U, t))
        w_mean = w.reshape(-1, w.shape[-1]).mean(dim=0)   # [K]
        w_mean = w_mean.view(1, 1, -1).expand_as(w)
        return w, w_mean
