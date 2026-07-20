"""gates.py — gating networks with anti-collapse countermeasures.

Everything here exists because of one empirical fact: naive learnable
gates (in particular learnable-center/learnable-sigma Gaussian bumps)
collapse to a single dominant expert.  The countermeasures baked in:

1. **Uniform start** — the final gate layer is zero-initialised, so at
   epoch 0 every expert receives identical weight (and hence identical
   gradient).  No expert can win before the others have learned anything.

2. **Temperature** — logits are divided by a temperature ``T`` stored as
   a buffer.  Train with warm T (soft blend) and anneal down via the
   ``GateTemperatureAnnealer`` callback in ``edm_v1_2.py``.

3. **Logit cap** — logits pass through ``cap * tanh(logit / cap)``, which
   bounds softmax saturation and guarantees a gradient floor for losing
   experts (max possible weight ratio is exp(2*cap/T)).

4. **Importance (load-balance) loss** — ``importance_from_weights``
   returns the batch-mean expert usage; penalising its deviation from
   uniform (done in ``edm_v1_2.py`` through a graph probe node) is the
   Shazeer-style importance loss.  This is the single most effective
   anti-dominance tool and was absent from the previous learnable-bump
   experiments.

5. **Sigma floor** (RBF gate only) — ``sigma = softplus(.) + sigma_min``
   prevents bumps from shrinking to spikes.

Note on collapse-as-diagnosis: if the balance loss merely redistributes
weights while validation error stays flat, the data may genuinely have a
single regime — that is evidence for the gateless LoRA variant, not for
harder regularisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _kmeans(
    x: torch.Tensor, k: int, n_iter: int = 50, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny torch-only k-means (k-means++ init + Lloyd iterations).

    Kept dependency-free so ``gates.py`` stays torch-only; sklearn is used
    elsewhere (Utils) but not imported into the model core.  Returns
    ``(labels [N], centers [k, D])``.
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    N = x.shape[0]
    # k-means++ seeding: first center random, rest by squared-distance prob.
    first = torch.randint(N, (1,), generator=g, device=x.device)
    centers = x[first]
    for _ in range(1, k):
        d2 = torch.cdist(x, centers).min(dim=1).values ** 2
        probs = d2 / d2.sum().clamp_min(1e-12)
        nxt = torch.multinomial(probs, 1, generator=g)
        centers = torch.cat([centers, x[nxt]], dim=0)

    labels = torch.full((N,), -1, dtype=torch.long, device=x.device)
    for _ in range(n_iter):
        new_labels = torch.cdist(x, centers).argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if m.any():  # leave empty clusters where they are (rare with ++ init)
                centers[j] = x[m].mean(dim=0)
    return labels, centers


class RBFGate(nn.Module):
    """RBF gate over gate features with a hard sigma floor.

    ``w = softmax( -0.5 * ||(f - mu_k)/sigma_k||^2 / T )``

    High-risk variant (kept for interpretability); use only with the
    full countermeasure stack: sigma floor here + temperature here +
    importance loss in the training objective.
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        sigma_min: float = 0.25,
        temperature_init: float = 2.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.sigma_min = sigma_min
        self.centers = nn.Parameter(torch.randn(n_experts, in_dim) * 0.5)
        self.log_sigma = nn.Parameter(torch.zeros(n_experts, in_dim))
        self.register_buffer("temperature", torch.tensor(float(temperature_init)))
        # Fixed feature projection (identity unless set by the per-trajectory
        # seeding path).  Buffer, not parameter: never trained, saved with
        # the state dict so seeding and runtime always agree.
        self.register_buffer("proj", torch.eye(in_dim))
        self.to(device)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_sigma) + self.sigma_min

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [..., in_dim]
        feats = feats @ self.proj.T                        # identity by default
        diff = feats.unsqueeze(-2) - self.centers          # [..., K, in_dim]
        d2 = ((diff / self.sigma) ** 2).sum(dim=-1)        # [..., K]
        return torch.softmax(-0.5 * d2 / self.temperature.clamp_min(1e-3), dim=-1)

    @torch.no_grad()
    def _fit_identity_projection(self, feats: torch.Tensor) -> None:
        """Set ``proj`` to remove the feature directions spanned by the
        per-trajectory means — the directions that encode *which trajectory*
        a sample came from rather than *where in the period* it is.

        ``feats``: [N_traj, T, D].  Removes enough directions to cover 99%
        of the between-trajectory variance, but only where that spread is
        non-negligible next to the within-trajectory motion, and always
        leaves >= 2 dims.
        """
        D = feats.shape[-1]
        traj_mu = feats.mean(dim=1)                        # [N_traj, D]
        Mc = traj_mu - traj_mu.mean(dim=0)
        within_std = (feats - traj_mu.unsqueeze(1)).reshape(-1, D).std()
        _, S, Vt = torch.linalg.svd(Mc, full_matrices=False)
        energy = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-12)
        n_energy = int((energy < 0.99).sum().item()) + 1
        n_signif = int((S / max(1, Mc.shape[0]) ** 0.5 > 0.1 * within_std).sum().item())
        m = min(n_energy, n_signif, D - 2)
        if m > 0:
            V = Vt[:m]
            self.proj.copy_(torch.eye(D, device=feats.device) - V.T @ V)

    @torch.no_grad()
    def init_from_segments(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        sigma_scale: float = 1.0,
        project: bool = True,
        fold_traj_spread: bool = True,
    ) -> torch.Tensor:
        """Supervised seeding: expert k is centered on the features observed
        during the time steps labelled k.

        ``feats``: [N_traj, T, D] gate features (``gate_feature_samples``
        with ``flatten=False``); ``labels``: [T] ints in [0, K) assigning
        time steps to experts, or -1 for steps to ignore during fitting
        (they still fall to the nearest seeded center at runtime — use -1
        to skip unrepresentative stretches such as the from-rest history
        fill-up transient).  Unlike the k-means seed this encodes
        the regime layout you *want* — use it when the regimes you care
        about are weakly separated in feature space (e.g. pre- vs post-pulse
        steady states that differ only through slow history memory), where
        unsupervised clustering merges them.

        ``project=True`` also fits the trajectory-identity projection (see
        ``_fit_identity_projection``).  The projection is linear, so when
        trajectory means trace a *curved* arc through feature space (means
        varying nonlinearly with the sweep parameter), mid-sweep
        trajectories keep a residual identity offset; ``fold_traj_spread``
        widens each expert's sigmas by the between-trajectory spread of its
        own segment means, so every trajectory's version of a regime stays
        inside its expert's bump.  Returns the achieved hard (argmax)
        usage fractions on the sample — compare them to the label fractions
        to see how separable the requested segments actually are.
        """
        feats = feats.detach().to(self.centers.device, torch.float32)
        labels = torch.as_tensor(labels, device=feats.device).long()
        K = self.n_experts
        N, T, D = feats.shape
        if labels.shape != (T,):
            raise ValueError(f"labels must be shape [{T}], got {tuple(labels.shape)}")
        if int(labels.max()) >= K or int(labels.min()) < -1:
            raise ValueError(f"labels must be in [-1, {K}), got range "
                             f"[{int(labels.min())}, {int(labels.max())}]")

        self.proj.copy_(torch.eye(D, device=feats.device))
        if project:
            self._fit_identity_projection(feats)
        x = (feats @ self.proj.T).reshape(N * T, D)
        lab = labels.unsqueeze(0).expand(N, T).reshape(-1)

        global_std = x.std(dim=0).clamp_min(1e-3)
        mu = torch.zeros(K, D, device=x.device)
        sig = global_std.expand(K, -1).clone()
        xp = feats @ self.proj.T                      # [N, T, D] projected
        for k in range(K):
            pts = x[lab == k]
            if pts.shape[0] == 0:
                raise ValueError(f"segment {k} contains no time steps")
            mu[k] = pts.mean(dim=0)
            if pts.shape[0] > 1:
                sig[k] = pts.std(dim=0)
        sig = sig * sigma_scale
        if fold_traj_spread and N > 1:
            for k in range(K):
                seg_mu = xp[:, labels == k, :].mean(dim=1)      # [N, D]
                sig[k] = torch.sqrt(sig[k] ** 2 + seg_mu.var(dim=0, unbiased=False))
        sig = torch.maximum(sig, 0.1 * global_std)

        y = (sig - self.sigma_min).clamp_min(1e-2)
        self.centers.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.expm1(y)))

        # Achieved hard partition with the seeded parameters.
        w = self.forward(feats)                       # projects internally
        counts = torch.bincount(w.argmax(-1).reshape(-1), minlength=K).float()
        return (counts / counts.sum().clamp_min(1.0)).cpu()

    @torch.no_grad()
    def init_from_features(
        self, feats: torch.Tensor, sigma_scale: float = 1.0, n_iter: int = 50,
        max_samples: int = 20000,
    ) -> torch.Tensor:
        """Seed centers/sigmas by clustering an observed gate-feature sample.

        ``feats``: [N, in_dim] — the [h, u] features the gate will actually
        see (roll the history bank over the training forcing to build this;
        see ``RBFHistoryMoE_NODE.gate_feature_samples``).  Centers are the
        k-means centroids; each sigma is the per-dim within-cluster std
        (scaled by ``sigma_scale``), floored at 10% of the global per-dim
        spread so no bump collapses to a spike before training starts.

        ``feats`` may instead be [N_traj, T, in_dim].  Then the directions of
        feature space spanned by the per-trajectory means — the directions
        that encode *which trajectory* a sample came from rather than *where
        in the period* it is — are projected out of the gate metric (stored
        in the ``proj`` buffer, applied inside ``forward`` so seeding and
        runtime agree), and clustering happens in the projected space.  Use
        this when a plain seed turns the gate into a trajectory classifier
        (each trajectory pinned to one expert) rather than a within-period
        regime switcher.

        Returns the cluster population fractions (a data-side estimate of the
        expert usage the seed implies), so you can check it against the split
        you expect.
        """
        feats = feats.detach().to(self.centers.device, torch.float32)
        K = self.n_experts
        D = feats.shape[-1]

        self.proj.copy_(torch.eye(D, device=feats.device))
        if feats.ndim == 3:  # [N_traj, T, D] -> suppress trajectory identity
            self._fit_identity_projection(feats)
            x = feats.reshape(-1, D)
        else:
            x = feats
        x = x @ self.proj.T
        if x.shape[0] > max_samples:
            g = torch.Generator(device=x.device).manual_seed(0)
            x = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:max_samples]]

        labels, mu = _kmeans(x, K, n_iter=n_iter)

        global_std = x.std(dim=0).clamp_min(1e-3)
        sig = global_std.expand(K, -1).clone()
        for j in range(K):
            pts = x[labels == j]
            if pts.shape[0] > 1:
                sig[j] = pts.std(dim=0)
        sig = torch.maximum(sig * sigma_scale, 0.1 * global_std)

        # Invert sigma = softplus(log_sigma) + sigma_min  ->  log_sigma.
        y = (sig - self.sigma_min).clamp_min(1e-2)
        self.centers.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.expm1(y)))

        counts = torch.bincount(labels.clamp_min(0), minlength=K).float()
        return (counts / counts.sum().clamp_min(1.0)).cpu()
