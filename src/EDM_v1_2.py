"""EDM_v1_2 — EDM_v1_1 + the rbf_moe forcing-history gated NODE.

Single method: ``rbf_moe`` — K experts on [z, u], an RBF gate over causal
forcing-history features [h, u_gate], and a periodic, forcing-derived history
initialisation at every window start (``U_period``).

Gate-stability machinery:
    - periodic, forcing-derived history initialization at every window start,
    - the mechanical encoder predicts z only (never the gate history h),
    - configurable temperature scheduling and RBF sigma floor,
    - a non-uniform expert-usage prior wired through ``GateProbe``,
    - configurable gate-visible control channels to exclude constant parameters.

Config keys (all optional, defaults in NODE_DEFAULTS):
    training.Qs.GATEBALANCE   expert-usage-prior weight     (default 0)
    training.Qs.GATEANCHOR    regime-anchor weight          (default 0)
    training.Qs.GATEOVERLAP   regime-overlap weight         (default 0)
    training.Qs.EXPERTDIVERSITY expert-diversity weight     (default 0)
    model.n_experts           experts                       (default 3)
    model.n_hist              history features              (default 8)
    model.gate_temperature    initial gate temperature      (default 3.0)
    model.gate_t_min          annealed floor                (default 1.0)
    model.gate_anneal_gamma   per-epoch decay               (default 1.0)
    model.gate_lr_mult        LR multiplier for the gating pathway after
                              data-driven RBF seeding       (default 0.1)
    model.gate_sigma_scale    scale on within-cluster std   (default 1.25)
    model.gate_sigma_min      hard floor on RBF sigmas      (default 0.10)
    model.gate_control_indices instantaneous controls visible to the gate
    model.gate_balance_target optional non-uniform usage target
    model.gate_regime_boundaries anchor windows for GATEANCHOR/GATEOVERLAP

Usage:
    from EDM_v1_2 import EDM
    model = EDM(A_Mat, pinv_Theta, dt, t_max, config_global, device)
    model.build_model(method="rbf_moe")
    model.train_model(...)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.dataset import DictDataset
from torch.utils.data import DataLoader
from neuromancer.constraint import variable
from neuromancer.dynamics import integrators
from neuromancer.loggers import BasicLogger
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.system import Node, System

from EDM import (
    CheckpointedRK4,
    NoiseLayer,
    ResamplingCallback,
    decoder,
    encoder,
    full_space,
)
from EDM_v1_1 import EDM as _EDMv11
from trainer import Trainer

from gated_nodes import GateProbe, RBFHistoryMoE_NODE


def _device_is_cuda(device: str | torch.device) -> bool:
    """Whether ``device`` is an available CUDA device.

    Pinned host memory is a CUDA transfer optimization.  Enabling it for an
    MPS/XPU/CPU run is at best pointless and, on some PyTorch builds, fails
    while constructing the data loader.
    """
    return torch.device(device).type == "cuda" and torch.cuda.is_available()

NODE_DEFAULTS: dict[str, Any] = {
    "n_experts": 3,
    "n_hist": 8,
    "gate_temperature": 3.0,
    "gate_t_min": 1.0,
    "gate_anneal_gamma": 1.0,
    "gate_balance_Q": 0.0,
    "gate_balance_target": None,
    "gate_lr_mult": 0.1,
    "gate_sigma_scale": 1.25,
    "gate_sigma_min": 0.10,
    "gate_control_indices": [0],
    "use_phase_in_gate": True,
    "gate_phase_harmonics": 1,
    "gate_regime_boundaries": [0.47, 0.53],
    "expert_max_cosine_similarity": 0.95,
}


class SliceDecoder(nn.Module):
    """Decode only the dynamics part z of the augmented latent [z | h].

    The history features h are inputs to the gate/adapters, not part of
    the reconstruction target — letting the decoder read them would let
    the AE losses repurpose h and destroy its meaning.
    """

    def __init__(self, dec: nn.Module, lat_space: int) -> None:
        super().__init__()
        self.dec = dec
        self.lat_space = lat_space

    def forward(self, z_aug):  # noqa: ANN001, ANN201
        return self.dec(z_aug[..., : self.lat_space])


class GateLabCallback(ResamplingCallback):
    """ResamplingCallback + gate-temperature annealing + usage report.

    Annealing: T(epoch) = max(t_min, T0 * gamma^epoch).  Warm start keeps
    every expert in play early; the slow cool-down lets specialisation
    emerge only after all experts have learned something.
    """

    def __init__(
        self,
        edm: "EDM",
        device: str = "cpu",
        every: int = 10,
    ) -> None:
        super().__init__(edm=edm, device=device)
        self.every = max(1, int(every))

    def end_epoch(self, trainer, output: dict) -> None:  # noqa: ANN001
        super().end_epoch(trainer, output)
        edm = self.edm
        epoch = trainer.current_epoch

        gate = getattr(edm, "_gate_module", None)
        if gate is not None:
            t0 = edm._gate_cfg["gate_temperature"]
            t_min = edm._gate_cfg["gate_t_min"]
            gamma = edm._gate_cfg["gate_anneal_gamma"]
            new_T = max(t_min, t0 * (gamma ** epoch))
            gate.temperature.fill_(new_T)

            if epoch % self.every == 0:
                probe = getattr(edm, "_gate_probe", None)
                usage = getattr(probe, "last_mean", None) if probe is not None else None
                usage_str = (
                    "usage=" + "/".join(f"{v:.3f}" for v in usage.tolist())
                    if usage is not None
                    else "usage=n/a (no batch seen yet)"
                )
                print(f"  [GateLab] epoch {epoch:4d}  T={new_T:.3f}  {usage_str}")


class _CachingGateProbe(GateProbe):
    """GateProbe that caches the last batch-mean usage for the callback."""

    def forward(self, LS, U, t):  # noqa: ANN001, ANN201
        w, w_mean = super().forward(LS, U, t)
        self.last_mean = w_mean[0, 0].detach().cpu()
        return w, w_mean


class _UsageTarget(nn.Module):
    """Broadcast a configurable expert-usage prior to the probe shape."""

    def __init__(
        self, n_experts: int, target=None, device: str | torch.device = "cpu"
    ) -> None:  # noqa: ANN001
        super().__init__()
        if target is None:
            target = torch.full((n_experts,), 1.0 / n_experts, device=device)
        # ``register_buffer`` has no ``device`` argument.  Put the tensor on
        # the requested device before registering it so regular module moves
        # (including accelerator placement) continue to work correctly.
        self.register_buffer("target", self._validate(target, n_experts, device))

    @staticmethod
    def _validate(
        target, n_experts: int, device: str | torch.device
    ) -> torch.Tensor:  # noqa: ANN001
        value = torch.as_tensor(target, dtype=torch.float32, device=device).flatten()
        if value.numel() != n_experts or torch.any(value < 0) or float(value.sum()) <= 0:
            raise ValueError(
                f"gate balance target must contain {n_experts} non-negative values, "
                f"got {value.tolist()}"
            )
        return value / value.sum()

    @torch.no_grad()
    def set_target(self, target) -> None:  # noqa: ANN001
        # Do not cache the construction device: ``Problem.to(...)`` and
        # distributed/accelerate placement can move this buffer later.
        value = self._validate(target, self.target.numel(), self.target.device)
        self.target.copy_(value)

    def forward(self, w_mean):  # noqa: ANN001, ANN201
        shape = [1] * (w_mean.ndim - 1) + [self.target.numel()]
        return (w_mean * 0.0) + self.target.view(*shape)


class _GateSpecializationProbe(nn.Module):
    """Phase-supervised routing and local gate-overlap penalties."""

    def __init__(
        self,
        t_max: float,
        n_experts: int,
        boundaries,
        eps: float = 1.0e-8,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        boundaries_t = torch.as_tensor(boundaries, dtype=torch.float32, device=device).flatten()
        if boundaries_t.numel() != n_experts - 1:
            raise ValueError(
                f"Expected {n_experts - 1} regime boundaries for {n_experts} experts, "
                f"received {boundaries_t.tolist()}."
            )
        if torch.any(boundaries_t <= 0.0) or torch.any(boundaries_t >= 1.0):
            raise ValueError("gate_regime_boundaries must lie strictly inside (0, 1).")
        if boundaries_t.numel() > 1 and torch.any(boundaries_t[1:] <= boundaries_t[:-1]):
            raise ValueError("gate_regime_boundaries must be strictly increasing.")
        self.t_max = float(t_max)
        self.n_experts = int(n_experts)
        self.eps = float(eps)
        self.register_buffer("boundaries", boundaries_t)

    def forward(
        self,
        gate_w: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = t[..., 0] if t.ndim == gate_w.ndim and t.shape[-1] == 1 else t
        phase = torch.remainder(phase, self.t_max) / self.t_max
        # The probe may be used outside a fully moved ``Problem`` (for
        # example during diagnostics), so make the comparison buffer follow
        # the runtime tensor as well as normal module ``.to(...)`` moves.
        boundaries = self.boundaries.to(device=phase.device, dtype=phase.dtype)
        labels = torch.bucketize(phase.contiguous(), boundaries)

        selected = gate_w.gather(-1, labels.unsqueeze(-1)).clamp_min(self.eps)
        anchor_root = torch.sqrt(-torch.log(selected) + self.eps)

        overlap = 1.0 - gate_w.square().sum(dim=-1, keepdim=True)
        overlap_root = torch.sqrt(overlap.clamp_min(0.0) + self.eps)
        return anchor_root, overlap_root


class _ExpertDiversityProbe(nn.Module):
    """Penalize expert vector fields that point in nearly identical directions."""

    def __init__(
        self,
        node: nn.Module,
        max_cosine_similarity: float = 0.95,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if not -1.0 < max_cosine_similarity < 1.0:
            raise ValueError("expert_max_cosine_similarity must be in (-1, 1).")
        object.__setattr__(self, "node", node)
        self.max_cosine_similarity = float(max_cosine_similarity)
        self.eps = float(eps)

    def forward(self, LS: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        T = U.shape[1]
        z = LS[:, :T, : self.node.lat_space].detach()
        expert_input = torch.cat([z, U.detach()], dim=-1)

        expert_outputs = torch.stack(
            [expert(expert_input) for expert in self.node.experts],
            dim=-2,
        )
        normalized = F.normalize(expert_outputs, p=2, dim=-1, eps=self.eps)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))

        K = cosine.shape[-1]
        pair_idx = torch.triu_indices(K, K, offset=1, device=cosine.device)
        pairwise = cosine[..., pair_idx[0], pair_idx[1]]
        penalty = torch.relu(pairwise - self.max_cosine_similarity).square()
        return torch.sqrt(penalty.mean(dim=-1, keepdim=True) + self.eps)


class _ZeroLike(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(value)


class _PeriodicStateInitializer(nn.Module):
    """Append a forcing-derived periodic history state to encoded ``z0``.

    The encoder never predicts ``h0``.  Consequently dropout/noise in the
    mechanical-state encoder cannot corrupt the gate state, and every training
    window uses the causal history corresponding to its actual start phase.
    """

    def __init__(self, node: nn.Module, dt: float) -> None:
        super().__init__()
        # The history bank is already registered inside NODE_System.  A
        # non-registered reference avoids duplicate state-dict entries while
        # retaining full autograd connectivity.
        object.__setattr__(self, "history", node.history)
        self.dt = float(dt)

    def forward(self, z0: torch.Tensor, U_period: torch.Tensor) -> torch.Tensor:
        h0 = self.history.periodic_initial_state(U_period, self.dt).to(z0.dtype)
        while h0.ndim < z0.ndim:
            h0 = h0.unsqueeze(1)
        return torch.cat([z0, h0], dim=-1)


def _build_periodic_windows_with_context(X, U, t, nBPP: int, lMB: int):  # noqa: ANN001, ANN201
    """Build cyclic windows and the full forcing cycle aligned to each start.

    ``U_period[i,0]`` is the control at the first step of window ``i``.  The
    periodic history initializer can therefore compute the exact history state
    immediately before that control is applied.
    """
    n_param, nt, nx = X.shape
    if U.shape[:2] != (n_param, nt):
        raise ValueError(f"X and U leading shapes must agree, got {X.shape} and {U.shape}")
    if lMB > nt:
        raise ValueError(f"lMB={lMB} cannot exceed periodic trajectory length nt={nt}")

    starts = np.random.randint(0, nt, size=(n_param, nBPP), dtype=np.int64)
    offsets = np.arange(lMB, dtype=np.int64)[None, None, :]
    window_idx = (starts[..., None] + offsets) % nt
    period_offsets = np.arange(nt, dtype=np.int64)[None, None, :]
    period_idx = (starts[..., None] + period_offsets) % nt
    param_idx = np.arange(n_param, dtype=np.int64)[:, None, None]

    winX = X[param_idx, window_idx, :].reshape(n_param * nBPP, lMB, nx)
    winU = U[param_idx, window_idx, :].reshape(n_param * nBPP, lMB, U.shape[-1])
    wint = t[window_idx].reshape(n_param * nBPP, lMB, t.shape[-1])
    U_period = U[param_idx, period_idx, :].reshape(n_param * nBPP, nt, U.shape[-1])
    return (
        np.ascontiguousarray(winX),
        np.ascontiguousarray(winU),
        np.ascontiguousarray(wint),
        np.ascontiguousarray(U_period),
    )


class EDM(_EDMv11):
    """EDM_v1_1 + the rbf_moe forcing-history gated NODE."""

    def __init__(self, A_mat, pinv_Theta, dt, t_max, config, device="cpu", mesh_json_path=None) -> None:
        super().__init__(A_mat, pinv_Theta, dt, t_max, config, device, mesh_json_path=mesh_json_path)
        self._gate_module = None
        self._gate_node = None
        self._gate_probe = None
        self._gate_target_module = None
        self._gate_cfg = dict(NODE_DEFAULTS)
        mdl_cfg = config.get("model", {})
        for key in ("n_experts", "n_hist", "gate_temperature",
                    "gate_t_min", "gate_anneal_gamma", "gate_lr_mult",
                    "gate_sigma_scale", "gate_sigma_min", "gate_balance_target",
                    "gate_control_indices", "use_phase_in_gate",
                    "gate_phase_harmonics", "gate_regime_boundaries",
                    "expert_max_cosine_similarity"):
            if key in mdl_cfg:
                self._gate_cfg[key] = mdl_cfg[key]

    # ─────────────────────────────────────────────────────────────────
    #  build_model
    # ─────────────────────────────────────────────────────────────────
    def build_model(
        self,
        method: str = "rbf_moe",
        checkpoint_rk4: bool = False,
    ) -> None:
        if method != "rbf_moe":
            raise ValueError(
                f"This EDM variant supports only method='rbf_moe', got {method!r}."
            )

        cfg = self._gate_cfg

        # ── NODE variant ───────────────────────────────────────────
        common = dict(
            lat_space=self.lat_space,
            n_control=self.n_control,
            n_layers=self.n_NODE_layers,
            n_units=self.n_NODE_units,
            device=self.device,
        )
        NODE_init = RBFHistoryMoE_NODE(
            dt=self.dt, t_max=self.t_max,
            n_experts=cfg["n_experts"], n_hist=cfg["n_hist"],
            sigma_min=cfg["gate_sigma_min"],
            gate_control_indices=cfg["gate_control_indices"],
            use_phase_in_gate=cfg["use_phase_in_gate"],
            gate_phase_harmonics=cfg["gate_phase_harmonics"],
            temperature_init=cfg["gate_temperature"], **common,
        )

        self.method = method
        self._gate_module = getattr(NODE_init, "gate", None)
        self._gate_node = NODE_init
        print(f"[EDM_v1_2] Using NODE variant: {NODE_init.__class__.__name__}")

        augment = getattr(NODE_init, "augment_state", False)
        lat = self.lat_space

        # The encoder represents only the mechanical latent z.  For augmented
        # methods h0 is computed from the aligned forcing cycle, never inferred
        # from x0 and never exposed to encoder dropout or latent noise.
        Encoder_init = encoder(self.n_sparse, lat, self.Encoder_hsizes, self.device)
        dec_core = decoder(self.n_sparse, lat, self.Decoder_hsizes, self.device)
        rollout_decoder = SliceDecoder(dec_core, lat) if augment else dec_core

        fu = nn.Identity()
        if checkpoint_rk4:
            fxRK4 = CheckpointedRK4(NODE_init, h=self.dt).to(self.device)
        else:
            fxRK4 = integrators.RK4(NODE_init, h=self.dt).to(self.device)

        noise_init = NoiseLayer(std=self.noise_std, device=self.device)
        encoder_x0 = Node(Encoder_init, ["x0"], ["LS_z0"], name="Encoder_x")
        noiseBlock = Node(noise_init, ["LS_z0"], ["LS_z"], name="Noise")
        fu_node = Node(fu, ["U"], ["U"], name="Control_Encoder")

        if augment:
            init_state = Node(
                _PeriodicStateInitializer(NODE_init, self.dt),
                ["LS_z", "U_period"], ["LS_x"], name="History_Init",
            )
        else:
            init_state = Node(nn.Identity(), ["LS_z"], ["LS_x"], name="State_Init")

        model = Node(fxRK4, ["LS_x", "U", "t"], ["LS_x"], name="NODE")
        decoder_x = Node(rollout_decoder, ["LS_x"], ["x_hat"], name="Decoder_x")
        encoder_FX = Node(Encoder_init, ["X"], ["LS_X"], name="Encoder_X")
        decoder_FX = Node(dec_core, ["LS_X"], ["X_hat"], name="Decoder_X")
        dynamics_model = System([model], name="NODE_System", nsteps=self.lMB)

        problem_nodes = [
            encoder_x0, encoder_FX, noiseBlock, fu_node, init_state,
            dynamics_model, decoder_x, decoder_FX,
        ]

        # ── losses in SPARSE space (inherited philosophy) ──────────
        X_true = variable("X")
        X_ae = variable("X_hat")
        X_aen = variable("x_hat")[:, :-1, :]

        # Latent consistency only on the dynamics part z: the encoder's h
        # estimate and the rolled-out h are both model quantities; forcing
        # them to agree early destabilises the history bank.
        ls_ae = variable("LS_X")[:, :, :lat]
        ls_aen = variable("LS_x")[:, :, :lat]

        FDt_true = X_true[:, 2:, :] - X_true[:, 1:-1, :]
        FDt_pred = X_aen[:, 2:, :] - X_aen[:, 1:-1, :]

        aenode_loss = self.Qs["AENODE"] * (X_aen == X_true) ^ 2
        aenode_loss.name = "AENODE loss"
        ae_loss = self.Qs["AE"] * (X_ae == X_true) ^ 2
        ae_loss.name = "AE loss"
        onestep_loss = self.Qs["ONESTEP"] * (X_aen[:, 1, :] == X_true[:, 1, :]) ^ 2
        onestep_loss.name = "One Step loss"
        laststep_loss = self.Qs["LASTSTEP"] * (X_aen[:, -1, :] == X_true[:, -1, :]) ^ 2
        laststep_loss.name = "Last Step loss"
        ls_loss = self.Qs["LS"] * (ls_ae == ls_aen[:, :-1, :]) ^ 2
        ls_loss.name = "Latent Space Loss"
        tdf_loss = self.Qs["TEMPORALDIFF"] * (FDt_pred == FDt_true) ^ 2
        tdf_loss.name = "Temporal Diff Loss"

        objectives = [
            aenode_loss, ae_loss, onestep_loss,
            laststep_loss, ls_loss, tdf_loss,
        ]

        # ── expert-usage prior ─────────────────────────────────────
        probe = _CachingGateProbe(NODE_init, detach_features=True)
        self._gate_probe = probe
        probe_node = Node(
            probe, ["LS_x", "U", "t"], ["gate_w", "gate_w_mean"], name="GateProbe",
        )
        target_module = _UsageTarget(cfg["n_experts"], cfg["gate_balance_target"], device=self.device)
        self._gate_target_module = target_module
        target_node = Node(
            target_module, ["gate_w_mean"], ["gate_usage_target"], name="GateUsageTarget",
        )
        problem_nodes.extend([probe_node, target_node])

        Q_bal = float(self.Qs.get("GATEBALANCE", cfg["gate_balance_Q"]))
        balance_loss = Q_bal * (
            variable("gate_w_mean") == variable("gate_usage_target")
        ) ^ 2
        balance_loss.name = "Gate Usage Prior Loss"
        objectives.append(balance_loss)

        Q_anchor = float(self.Qs.get("GATEANCHOR", 0.0))
        Q_overlap = float(self.Qs.get("GATEOVERLAP", 0.0))
        Q_diversity = float(self.Qs.get("EXPERTDIVERSITY", 0.0))

        if Q_anchor > 0.0 or Q_overlap > 0.0:
            specialization_probe = _GateSpecializationProbe(
                t_max=self.t_max,
                n_experts=cfg["n_experts"],
                boundaries=cfg["gate_regime_boundaries"],
                device=self.device,
            )
            specialization_node = Node(
                specialization_probe,
                ["gate_w", "t"],
                ["gate_anchor_root", "gate_overlap_root"],
                name="GateSpecialization",
            )
            anchor_zero_node = Node(
                _ZeroLike(),
                ["gate_anchor_root"],
                ["gate_anchor_zero"],
                name="GateAnchorZero",
            )
            overlap_zero_node = Node(
                _ZeroLike(),
                ["gate_overlap_root"],
                ["gate_overlap_zero"],
                name="GateOverlapZero",
            )
            problem_nodes.extend(
                [specialization_node, anchor_zero_node, overlap_zero_node]
            )

            if Q_anchor > 0.0:
                anchor_loss = Q_anchor * (
                    variable("gate_anchor_root") == variable("gate_anchor_zero")
                ) ^ 2
                anchor_loss.name = "Gate Regime Anchor Loss"
                objectives.append(anchor_loss)

            if Q_overlap > 0.0:
                overlap_loss = Q_overlap * (
                    variable("gate_overlap_root") == variable("gate_overlap_zero")
                ) ^ 2
                overlap_loss.name = "Gate Overlap Loss"
                objectives.append(overlap_loss)

        if Q_diversity > 0.0:
            diversity_probe = _ExpertDiversityProbe(
                NODE_init,
                max_cosine_similarity=cfg["expert_max_cosine_similarity"],
            )
            diversity_node = Node(
                diversity_probe,
                ["LS_x", "U"],
                ["expert_diversity_root"],
                name="ExpertDiversity",
            )
            diversity_zero_node = Node(
                _ZeroLike(),
                ["expert_diversity_root"],
                ["expert_diversity_zero"],
                name="ExpertDiversityZero",
            )
            problem_nodes.extend([diversity_node, diversity_zero_node])
            diversity_loss = Q_diversity * (
                variable("expert_diversity_root")
                == variable("expert_diversity_zero")
            ) ^ 2
            diversity_loss.name = "Expert Output Diversity Loss"
            objectives.append(diversity_loss)

        target_str = "/".join(
            f"{v:.3f}" for v in target_module.target.detach().cpu().tolist()
        )
        print(
            f"[EDM_v1_2] Gate losses: balance={Q_bal}, "
            f"anchor={Q_anchor}, overlap={Q_overlap}, "
            f"expert_diversity={Q_diversity}; target={target_str}"
        )

        # ── CST spatial regulariser (identical to EDM_v1_1) ────────
        if self.gradient_module is not None:
            FS_init = full_space(self.n_sparse, self.A_mat, self.pinv_Theta, self.device)
            FS_true = Node(FS_init, ["X"], ["X_full"], name="FS_true")
            FS_pred = Node(FS_init, ["x_hat"], ["x_hat_full"], name="FS_pred")
            grad_true_node = Node(self.gradient_module, ["X_full"], ["X_grad"], name="Grad_X")
            grad_pred_node = Node(self.gradient_module, ["x_hat_full"], ["x_hat_grad"], name="Grad_x")
            problem_nodes.extend([FS_true, FS_pred, grad_true_node, grad_pred_node])

            G_true = variable("X_grad")
            G_pred = variable("x_hat_grad")[:, :-1, :]

            sparse_dim = 2 * self.n_sparse
            grad_dim = self.gradient_module.output_dim
            dim_scale = sparse_dim / max(grad_dim, 1)
            scaled_Q = self.Qs["SPATIALDIFF"] * dim_scale
            xdf_loss = scaled_Q * (G_pred == G_true) ^ 2
            xdf_loss.name = "Spatial Gradient Loss"
            print(
                f"[EDM_v1_2] CST gradient loss: grad_dim={grad_dim}, "
                f"sparse_dim={sparse_dim}, dim_scale={dim_scale:.4f}, "
                f"effective Q={scaled_Q:.6f}"
            )
            objectives.append(xdf_loss)
        else:
            CDx_true = X_true[:, :, 3:] - 2 * X_true[:, :, 2:-1] + X_true[:, :, 1:-2]
            CDx_pred = X_aen[:, :, 3:] - 2 * X_aen[:, :, 2:-1] + X_aen[:, :, 1:-2]
            xdf_loss = self.Qs["SPATIALDIFF"] * (CDx_pred == CDx_true) ^ 2
            xdf_loss.name = "Spatial Diff Loss"
            objectives.append(xdf_loss)

        loss = PenaltyLoss(objectives, [])
        self.problem = Problem(problem_nodes, loss)
        # Some auxiliary loss probes contain buffers but no parameters.  Move
        # the whole graph so those buffers always match the model device.
        self.problem.to(self.device)
        self.optimizer = torch.optim.Adam(self.problem.parameters(), lr=self.lr)
        self.problem.show()

    # ─────────────────────────────────────────────────────────────────
    #  init_rbf_gate_from_data — seed the RBF gate from the training
    #  forcing and put the whole gating pathway on a small LR.
    # ─────────────────────────────────────────────────────────────────
    def init_rbf_gate_from_data(
        self,
        ft_train,
        sigma_scale: float | None = None,
        gate_lr_mult: float | None = None,
        max_samples: int = 20000,
        per_trajectory: bool = False,
    ) -> torch.Tensor:
        """Seed the RBF gate from the forcing, then slow the gating pathway.

        Call AFTER ``build_model(method="rbf_moe")`` and BEFORE
        ``train_model``, passing the same ``ft_train`` forcing array.  Steps:

        1. Roll the (freshly-initialised) history bank over ``ft_train`` to
           get the [h, u] gate features (``gate_feature_samples``).
        2. K-means those features into ``n_experts`` clusters -> RBF centers;
           set each sigma from the within-cluster spread
           (``RBFGate.init_from_features``).
        3. Rebuild the optimizer with two groups: everything at the base LR,
           but the *gating pathway* (RBF centers/sigmas AND the history bank
           that produces the gate features) at ``base_lr * gate_lr_mult`` —
           so the seeded regime structure trains, but only very slowly.

        ``per_trajectory=True`` centers the features per trajectory before
        clustering, so the seed captures *within-period* regime structure
        instead of assigning whole trajectories to experts.  Use it when the
        default seed pins each trajectory to a single expert (check with the
        per-trajectory hard-usage table).

        Returns the cluster population fractions implied by the seed, so you
        can compare against the split you expect (e.g. ~0.45/0.10/0.45).
        """
        if self.method != "rbf_moe":
            raise ValueError(
                f"init_rbf_gate_from_data is only for method='rbf_moe' "
                f"(current method={self.method!r})."
            )
        node, gate = self._gate_node, self._gate_module
        if node is None or gate is None or self.problem is None:
            raise ValueError(
                "Call build_model(method='rbf_moe') before init_rbf_gate_from_data."
            )
        cfg = self._gate_cfg
        sigma_scale = cfg["gate_sigma_scale"] if sigma_scale is None else sigma_scale
        gate_lr_mult = cfg["gate_lr_mult"] if gate_lr_mult is None else gate_lr_mult

        # Use the live NODE device instead of the construction setting: the
        # graph may have been moved by ``Problem.to`` or an accelerator.
        node_device = next(node.parameters()).device
        U = torch.as_tensor(ft_train, dtype=torch.float32, device=node_device)
        feats = node.gate_feature_samples(
            U, max_samples=max_samples, flatten=not per_trajectory,
        )
        usage = gate.init_from_features(
            feats, sigma_scale=sigma_scale, max_samples=max_samples,
        )
        usage_str = "/".join(f"{v:.3f}" for v in usage.tolist())
        mode = "per-trajectory (within-period)" if per_trajectory else "pooled"
        print(
            f"[EDM_v1_2] RBF gate seeded ({mode}) from "
            f"{feats.reshape(-1, feats.shape[-1]).shape[0]} forcing-history "
            f"samples; cluster fractions = {usage_str}"
        )

        if self._gate_target_module is not None:
            self._gate_target_module.set_target(usage)
            print(
                "[EDM_v1_2] Gate usage prior updated from the seeded partition: "
                + "/".join(f"{v:.3f}" for v in usage.tolist())
            )
        self._set_gating_lr_groups(gate_lr_mult)
        return usage

    def _set_gating_lr_groups(self, gate_lr_mult: float) -> None:
        """Rebuild the optimizer with the gating pathway (RBF centers/sigmas
        + the history bank that generates the gate features) on
        ``base_lr * gate_lr_mult``; everything else stays at the base LR.
        ``ReduceLROnPlateau`` scales both groups proportionally, so the
        ratio holds for the whole run."""
        node, gate = self._gate_node, self._gate_module
        gating_params = list(gate.parameters()) + list(node.history.parameters())
        gating_ids = {id(p) for p in gating_params}
        other_params = [
            p for p in self.problem.parameters()
            if p.requires_grad and id(p) not in gating_ids
        ]
        gate_lr = self.lr * gate_lr_mult
        self.optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": self.lr},
                {"params": gating_params, "lr": gate_lr},
            ]
        )
        print(
            f"[EDM_v1_2] Gating pathway on reduced LR {gate_lr:.2e} "
            f"(= base {self.lr:.2e} x {gate_lr_mult}); "
            f"{len(gating_params)} gating tensors, {len(other_params)} others."
        )

    # ─────────────────────────────────────────────────────────────────
    #  init_rbf_gate_from_segments — supervised seeding: expert k owns
    #  the k-th time segment of the trajectory.
    # ─────────────────────────────────────────────────────────────────
    def init_rbf_gate_from_segments(
        self,
        ft_train,
        boundaries: tuple[float, ...] = (0.45, 0.55),
        windows: list[tuple[float, float] | list[tuple[float, float]]] | None = None,
        sigma_scale: float | None = None,
        gate_lr_mult: float | None = None,
        balance_target: tuple[float, ...] | list[float] | None = None,
        per_trajectory: bool = True,
        fold_traj_spread: bool = True,
    ) -> torch.Tensor:
        """Seed the RBF gate so expert k owns the k-th time segment.

        ``boundaries``: K-1 increasing fractions of the trajectory splitting
        it into K segments, assigned to experts in order.  The default
        ``(0.45, 0.55)`` gives expert 0 the first 45%, expert 1 the middle
        10%, expert 2 the last 45%.

        ``windows`` (overrides ``boundaries``): one entry per expert.  Each
        entry may be a single ``(start, end)`` interval or a list of intervals,
        allowing one background expert to own both the pre- and post-pulse
        portions, e.g. ``[[(0,.47),(.53,1)], [(0.47,.53)]]``.  Intervals use
        endpoint-excluded periodic fractions and may cover the entire cycle.

        Use this instead of ``init_rbf_gate_from_data`` when you already
        know the regime layout: unsupervised k-means merges regimes whose
        features barely differ (e.g. identical pre-/post-pulse steady
        states), while this places each expert's center on its segment's
        features directly.  Call after ``build_model(method='rbf_moe')``,
        before ``train_model``; also puts the gating pathway on
        ``base_lr * gate_lr_mult`` like ``init_rbf_gate_from_data``.

        Prints intended vs achieved usage — if they disagree badly, the
        segments are not separable in the gate's forcing-history features
        and no seeding can hold them apart.
        """
        if self.method != "rbf_moe":
            raise ValueError(
                f"init_rbf_gate_from_segments is only for method='rbf_moe' "
                f"(current method={self.method!r})."
            )
        node, gate = self._gate_node, self._gate_module
        if node is None or gate is None or self.problem is None:
            raise ValueError(
                "Call build_model(method='rbf_moe') before init_rbf_gate_from_segments."
            )
        cfg = self._gate_cfg
        K = cfg["n_experts"]
        sigma_scale = cfg["gate_sigma_scale"] if sigma_scale is None else sigma_scale
        gate_lr_mult = cfg["gate_lr_mult"] if gate_lr_mult is None else gate_lr_mult

        # Keep seed data colocated with the current NODE after any external
        # device placement.
        node_device = next(node.parameters()).device
        U = torch.as_tensor(ft_train, dtype=torch.float32, device=node_device)
        feats = node.gate_feature_samples(U, flatten=False)     # [N, T, D]
        T = feats.shape[1]
        t_frac = torch.arange(T, device=feats.device, dtype=feats.dtype) / T

        if windows is not None:
            if len(windows) != K:
                raise ValueError(f"windows must contain one entry per expert ({K}), got {windows}")
            wins: list[list[tuple[float, float]]] = []
            for entry in windows:
                if (
                    isinstance(entry, (tuple, list)) and len(entry) == 2
                    and all(isinstance(v, (int, float)) for v in entry)
                ):
                    group = [(float(entry[0]), float(entry[1]))]
                else:
                    group = [(float(a), float(b)) for a, b in entry]
                if not group or not all(0.0 <= a < b <= 1.0 for a, b in group):
                    raise ValueError(f"invalid expert window group: {entry}")
                wins.append(group)

            labels = torch.full((T,), -1, dtype=torch.long, device=feats.device)
            for k, group in enumerate(wins):
                for a, b in group:
                    mask = (t_frac >= a) & ((t_frac < b) if b < 1.0 else (t_frac <= b))
                    if torch.any(labels[mask] >= 0):
                        raise ValueError(f"expert windows overlap near ({a}, {b})")
                    labels[mask] = k
            seed_desc = f"windows {wins}"
        else:
            bounds = [float(b) for b in boundaries]
            if len(bounds) != K - 1 or bounds != sorted(bounds) or not all(
                0.0 < b < 1.0 for b in bounds
            ):
                raise ValueError(
                    f"boundaries must be {K - 1} increasing fractions in (0, 1), got {boundaries}"
                )
            b = torch.tensor(bounds, device=feats.device)
            labels = (t_frac.unsqueeze(1) > b.unsqueeze(0)).sum(dim=1)  # [T]
            seed_desc = f"segments {bounds}"

        achieved = gate.init_from_segments(
            feats, labels, sigma_scale=sigma_scale, project=per_trajectory,
            fold_traj_spread=fold_traj_spread,
        )
        lab_mask = labels >= 0
        intended = torch.bincount(labels[lab_mask], minlength=K).float() / max(1, int(lab_mask.sum()))
        w = gate(feats)                                          # [N, T, K]
        agree = (
            (w.argmax(-1)[:, lab_mask] == labels[lab_mask].unsqueeze(0)).float().mean().item()
        )
        fmt = lambda v: "/".join(f"{x:.3f}" for x in v.tolist())  # noqa: E731
        print(
            f"[EDM_v1_2] RBF gate seeded from {seed_desc}: "
            f"fitted-step usage {fmt(intended)}, achieved (all steps) {fmt(achieved)}, "
            f"fitted-step agreement {agree:.1%}"
        )
        usage_target = achieved if balance_target is None else torch.as_tensor(balance_target)
        if self._gate_target_module is not None:
            self._gate_target_module.set_target(usage_target)
            print(
                "[EDM_v1_2] Gate usage prior updated to "
                + fmt(self._gate_target_module.target.detach().cpu())
            )
        self._set_gating_lr_groups(gate_lr_mult)
        return achieved

    # ─────────────────────────────────────────────────────────────────
    #  Periodic data API — every window carries its aligned forcing cycle
    #  so h0 can be computed causally inside the differentiable graph.
    # ─────────────────────────────────────────────────────────────────
    def _build_train_loader(self) -> DataLoader:
        trainX, trainU, traint, U_period = _build_periodic_windows_with_context(
            self._raw_train_X, self._raw_train_U, self._raw_t, self.nBPP, self.lMB
        )
        tensors = [torch.from_numpy(a).to(dtype=torch.float32) for a in (trainX, trainU, traint, U_period)]
        if not self._train_use_pin:
            tensors = [a.to(device=self.device) for a in tensors]
        trainX_t, trainU_t, traint_t, U_period_t = tensors
        dataset = DictDataset(
            {
                "X": trainX_t, "x0": trainX_t[:, 0:1, :],
                "U": trainU_t, "t": traint_t, "U_period": U_period_t,
            },
            name="train",
        )
        return DataLoader(
            dataset, batch_size=self.nMB, collate_fn=dataset.collate_fn,
            shuffle=True, pin_memory=self._train_use_pin,
            num_workers=self._train_num_workers,
            persistent_workers=self._train_num_workers > 0,
        )

    def get_data(
        self, X_train, U_train, X_dev, U_dev, X_test, U_test, t, num_workers: int = 0
    ):  # noqa: ANN001, ANN201
        use_pin = _device_is_cuda(self.device)
        self._raw_train_X = X_train
        self._raw_train_U = U_train
        self._raw_t = t
        self._train_use_pin = use_pin
        self._train_num_workers = num_workers
        train_loader = self._build_train_loader()

        devX, devU, devt, dev_period = _build_periodic_windows_with_context(
            X_dev, U_dev, t, self.nBPP, self.lMB
        )
        dev_tensors = [torch.from_numpy(a).to(dtype=torch.float32) for a in (devX, devU, devt, dev_period)]
        if not use_pin:
            dev_tensors = [a.to(device=self.device) for a in dev_tensors]
        devX_t, devU_t, devt_t, dev_period_t = dev_tensors
        dev_dataset = DictDataset(
            {
                "X": devX_t, "x0": devX_t[:, 0:1, :],
                "U": devU_t, "t": devt_t, "U_period": dev_period_t,
            },
            name="val",
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=self.nMB, collate_fn=dev_dataset.collate_fn,
            shuffle=False, pin_memory=use_pin, num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        testX = torch.from_numpy(np.ascontiguousarray(X_test)).to(
            dtype=torch.float32, device=self.device
        )
        testU = torch.from_numpy(np.ascontiguousarray(U_test)).to(
            dtype=torch.float32, device=self.device
        )
        testt = torch.from_numpy(np.ascontiguousarray(t)).to(
            dtype=torch.float32, device=self.device
        )
        test_data = {
            "X": testX, "x0": testX[:, 0:1, :], "U": testU,
            "t": testt.repeat(X_test.shape[0], 1, 1),
            "U_period": testU,
        }
        return train_loader, dev_loader, test_data

    # ─────────────────────────────────────────────────────────────────
    #  train_model — same as base, but installs GateLabCallback for the
    #  new methods (annealing + usage report on top of resampling).
    # ─────────────────────────────────────────────────────────────────
    def train_model(
        self,
        output_paths: str,
        data_train,
        ft_train,
        data_dev,
        ft_dev,
        data_test,
        ft_test,
        t,
        use_accelerate: bool = False,
        mixed_precision: str = "no",
        compile_model: bool = False,
        num_workers: int = 0,
    ) -> Problem:
        train_loader, dev_loader, test_data = self.get_data(
            data_train, ft_train, data_dev, ft_dev, data_test, ft_test, t,
            num_workers=num_workers,
        )

        callbacker = GateLabCallback(edm=self, device=self.device)
        logger = BasicLogger(
            args=None, savedir=output_paths, verbosity=1,
            stdout=["val_loss", "train_loss"],
        )

        if self.problem is None:
            raise ValueError("Problem has to be initiated first (call build_model).")

        trainer = Trainer(
            problem=self.problem,
            train_data=train_loader,
            dev_data=dev_loader,
            test_data=test_data,
            optimizer=self.optimizer,
            logger=logger,
            patience=self.patience,
            warmup=self.warmup,
            epochs=self.n_epoch,
            eval_metric="val_loss",
            train_metric="train_loss",
            dev_metric="val_loss",
            test_metric="val_loss",
            lr_scheduler=self.lr_patience,
            device=self.device,
            callback=callbacker,
            clip=5,
            use_accelerate=use_accelerate,
            mixed_precision=mixed_precision,
            compile_model=compile_model,
            num_workers=num_workers,
            output_paths=output_paths,
        )

        best_model = trainer.train()
        self.problem.load_state_dict(best_model)
        return self.problem
