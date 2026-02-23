#%% EDM.py — continuous (no-jump) NODE variants
#   - AE + NODE with q-conditioned LPV Residuals
#   - Options: "lpv_resid", "rbf_lpv_resid", "lora_lpv_resid", "phase_norm", "softgate"
#   - Keeps neuromancer-style Problem/Node/System API you already use.
#   - No hard bins, no jump detectors.

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import functional as AF

from neuromancer.dataset import DictDataset
from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger

# local trainer (unchanged)
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()
utils_dir = os.path.abspath(os.path.join(script_dir, "..", "Utils"))
sys.path.append(utils_dir)
from trainer import Trainer, custom_callback


# -----------------------
#  Shared building blocks
# -----------------------

class noiseLayer(nn.Module):
    def __init__(self, std=0.005, device='cpu'):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        if self.training and self.std > 0.0:
            noise = torch.normal(0, self.std, size=x.size(), device=self.device)
            x = x + noise
        return x


class encoder(nn.Module):
    """
    Encodes full-state RS = [displacements, velocities] (size 2*n_sparse)
    into a latent vector (size lat_space).
    """
    def __init__(self, n_sparse, lat_space, E_hsizes, device='cpu'):
        super().__init__()
        self.block = blocks.MLP(
            2*n_sparse, lat_space, bias=True,
            linear_map=torch.nn.Linear, nonlin=torch.nn.SiLU,
            hsizes=E_hsizes
        ).to(device)
        self.lin_layer = nn.Linear(lat_space, lat_space, bias=True).to(device)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(p=0.01)

    def forward(self, x):
        # x: [B, T, 2*n_sparse] or [B, 2*n_sparse] (Node handles T vs no-T)
        out = self.block(x)
        out = self.drop(self.lin_layer(out))
        return self.act(out)


class decoder(nn.Module):
    """
    Decodes latent vector to RS. We keep the *linear* output head
    (no tanh on the final) since you reported better AE Frobenius with that.
    """
    def __init__(self, n_sparse, lat_space, D_hsizes, device='cpu'):
        super().__init__()
        self.block = blocks.MLP(
            lat_space, 2*n_sparse, bias=True,
            linear_map=torch.nn.Linear, nonlin=torch.nn.SiLU,
            hsizes=D_hsizes
        ).to(device)
        self.lin_layer = nn.Linear(2*n_sparse, 2*n_sparse, bias=True).to(device)
        self.drop = nn.Dropout(p=0.01)

    def forward(self, x):
        out = self.block(x)
        out = self.drop(self.lin_layer(out))
        return out  # linear head (no tanh)


class full_space(nn.Module):
    """
    Map from RS (sparse) to full space FS by C = (A @ pinv_Theta)^T.
    Keeps your existing mapping.
    """
    def __init__(self, n_sparse, A_Mat, pinv_Theta, device):
        super().__init__()
        self.device = device
        # [n_full, n_sparse] then unsqueeze to broadcast over batch/time
        self.C = torch.tensor(A_Mat @ pinv_Theta, dtype=torch.float32, device=self.device).T.unsqueeze(0)
        self.n_sparse = n_sparse

    def forward(self, RS):
        RS1 = RS[:, :, :self.n_sparse]
        RS2 = RS[:, :, self.n_sparse:]
        FS1 = RS1 @ self.C
        FS2 = RS2 @ self.C
        return torch.cat([FS1, FS2], dim=-1)


# --------------------------------
#  Continuous time-gated base NODE
# --------------------------------
class softgate_NODE(nn.Module):
    """
    Time-gated (smooth) multi-expert NODE:
        f(x,u,t) = w1(t) f1([x,u]) + w2(t) f2([x,u]) + w3(t) f3([x,u])
    No hard gates, fully continuous in t; keeps in/out_features for RK4 wrapper.
    """
    def __init__(self, lat_space, n_control, n_layers, n_units, dt, gate_config, device='cpu'):
        super().__init__()
        self.lat_space = lat_space
        self.n_control = n_control
        self.in_features = lat_space + n_control
        self.out_features = lat_space
        self.dropout = nn.Dropout(p=0.05)
        self.t_max = 1.0  # assume t normalized 0..1 by dataloader
        self.sigma = 0.08

        if gate_config == "narrow":
            centers = torch.tensor([0.35, 0.50, 0.65], dtype=torch.float32)
        else:
            centers = torch.tensor([0.20, 0.50, 0.80], dtype=torch.float32)
        self.register_buffer("centers", centers.to(device))

        hs = [n_units] * n_layers
        self.pre  = blocks.MLP(self.in_features, lat_space, True, torch.nn.Linear, torch.nn.ReLU, hsizes=hs).to(device)
        self.mid  = blocks.MLP(self.in_features, lat_space, True, torch.nn.Linear, torch.nn.ReLU, hsizes=hs).to(device)
        self.post = blocks.MLP(self.in_features, lat_space, True, torch.nn.Linear, torch.nn.ReLU, hsizes=hs).to(device)

    def forward(self, x, u, t):
        # t: [B,1]; normalize if needed
        t_norm = t.clamp(0.0, 1.0)
        diff = (t_norm - self.centers).abs()
        weights = torch.softmax(- (diff**2) / (2 * self.sigma**2), dim=1)

        inp = torch.cat([x, u], dim=1)
        o1 = self.dropout(self.pre(inp))
        o2 = self.dropout(self.mid(inp))
        o3 = self.dropout(self.post(inp))
        # weights: [B,3], outputs: [B,L] — broadcast multiply
        return (weights[:, :1] * o1) + (weights[:, 1:2] * o2) + (weights[:, 2:3] * o3)


# ------------------------------------
#  Phase-normalized smooth time-gating
# ------------------------------------
class PhaseNorm_NODE(nn.Module):
    """
    Continuous, no-hard-switch phase gate + amplitude rescale.
    NOTE: Provided for completeness; use "lpv_resid" / "rbf_lpv_resid" as your main paths.
    """
    def __init__(self, lat_space, n_control, n_layers, n_units,
                 n_q=1, gate_config="regular", device="cpu", sigma=0.08):
        super().__init__()
        self.lat_space = lat_space
        self.n_control = n_control
        self.in_features = lat_space + n_control
        self.out_features = lat_space
        self.n_q = n_q
        self.sigma = sigma

        if gate_config == "narrow":
            centers = torch.tensor([0.35, 0.50, 0.65], dtype=torch.float32)
        else:
            centers = torch.tensor([0.20, 0.50, 0.80], dtype=torch.float32)
        self.register_buffer("centers", centers.to(device))

        hs = [n_units]*n_layers
        self.core = blocks.MLP(self.in_features, lat_space, True, torch.nn.Linear, nn.SiLU, hsizes=hs).to(device)
        # learn small scale/shift on latent derivative
        self.amp = nn.Sequential(nn.Linear(lat_space, lat_space), nn.Tanh()).to(device)
        self.bias = nn.Sequential(nn.Linear(lat_space, lat_space), nn.Tanh()).to(device)

    def forward(self, x, u, t):
        t_norm = t.clamp(0.0, 1.0)
        diff = (t_norm - self.centers).abs()
        w = torch.softmax(- (diff**2) / (2 * self.sigma**2), dim=1)  # [B,3]
        # collapse to a single weight by projecting onto [1,2,3] (smooth phase cue)
        phase_scalar = (w @ torch.tensor([1.0, 2.0, 3.0], device=w.device).unsqueeze(1)).squeeze(1) / 3.0
        inp = torch.cat([x, u], dim=1)
        h = self.core(inp)
        return (1.0 + 0.3*phase_scalar.unsqueeze(1)) * self.amp(h) + self.bias(h)


# -------------------------------------------------
#  LPV Residual family (no hard bins, no jumps)
# -------------------------------------------------

class LPVResidual_NODE(nn.Module):
    """
    q-conditioned linear core + residual MLP:
       f(x,u,t) = A(q) x + B(q) u + c(q) + r([x,u,t], q)
    """
    def __init__(self, lat_space, n_control, n_layers, n_units, n_q=1,
                 device="cpu", A_norm_cap=0.9, res_hmult=0.5):
        super().__init__()
        self.lat_space = lat_space
        self.n_control = n_control
        self.in_features = lat_space + n_control
        self.out_features = lat_space
        self.n_q = n_q
        self.A_norm_cap = A_norm_cap

        L, N = lat_space, n_control
        hid = max(32, min(128, 2*n_units))
        out_dim = L*L + L*N + L
        self.q2ABC = nn.Sequential(
            nn.Linear(n_q, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, out_dim)
        ).to(device)

        res_units = max(16, int(res_hmult*n_units))
        hsizes = [res_units]*n_layers
        self.residual = blocks.MLP(L + N + 1 + n_q, L, True, nn.Linear, nn.SiLU, hsizes=hsizes).to(device)
        self.res_gain = nn.Parameter(torch.tensor(0.1)).to(device)

    def _split_ABC(self, vec):
        L, N = self.lat_space, self.n_control
        A_flat = vec[:, :L*L]
        B_flat = vec[:, L*L:L*L + L*N]
        c      = vec[:, L*L + L*N:]
        A = A_flat.view(-1, L, L)
        B = B_flat.view(-1, L, N)
        return A, B, c

    def forward(self, x, u, t):
        # last n_q dims of U are the parameters q (your existing convention)
        q = u[:, -self.n_q:] if self.n_q > 0 else torch.zeros(x.size(0), 1, device=x.device)
        abc = self.q2ABC(q)
        A, B, c = self._split_ABC(abc)

        # A spectral normalization clamp (optional, lightweight)
        if self.A_norm_cap is not None:
            # quick Fro norm based clip
            A_norm = torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1), keepdim=True)  # [B,1,1]
            scale = torch.clamp(self.A_norm_cap / (A_norm + 1e-6), max=1.0)
            A = A * scale

        core = torch.einsum('bij,bj->bi', A, x) + torch.einsum('bij,bj->bi', B, u) + c
        rinp = torch.cat([x, u, t, q], dim=1)
        r = self.residual(rinp)
        return core + self.res_gain * r


class RBF_LPVResidual_NODE(nn.Module):
    """
    RBF mixture over K LPV vertices:
       A(q) = sum_k w_k(q) A_k,  same for B, c.  w_k: RBF over q.
       + small residual r([x,u,t], q)
    Smooth in q, no hard bins.
    """
    def __init__(self, lat_space, n_control, n_layers, n_units, n_q=1,
                 K=6, device="cpu", res_hmult=0.5):
        super().__init__()
        self.lat_space = lat_space
        self.n_control = n_control
        self.in_features = lat_space + n_control
        self.out_features = lat_space
        self.n_q = n_q
        self.K = K

        L, N = lat_space, n_control
        out_dim = L*L + L*N + L      # per-vertex params: A_k, B_k, c_k
        hid = max(64, 2*n_units)

        # centers and widths for RBF in q
        self.q_centers = nn.Parameter(torch.randn(K, n_q) * 0.25)
        self.q_logsigma = nn.Parameter(torch.zeros(K, n_q))

        # A_k, B_k, c_k parametrized by a small MLP of anchor index embedding
        self.vertex_embed = nn.Embedding(K, 32)
        self.vert_mlp = nn.Sequential(
            nn.Linear(32, hid), nn.SiLU(),
            nn.Linear(hid, out_dim)
        ).to(device)

        # residual conditioned on q
        res_units = max(16, int(res_hmult*n_units))
        hsizes = [res_units]*n_layers
        self.residual = blocks.MLP(L + N + 1 + n_q, L, True, nn.Linear, nn.SiLU, hsizes=hsizes).to(device)
        self.res_gain = nn.Parameter(torch.tensor(0.1)).to(device)


    def _rbf_weights(self, q):
        # q: [B, n_q] -> weights [B, K]
        # Gaussian RBF with learned anisotropic sigma per center
        # dist_k = sum_j ((q_j - mu_kj)/sigma_kj)^2
        B = q.size(0)
        q_exp = q.unsqueeze(1).expand(B, self.K, self.n_q)      # [B,K,n_q]
        mu = self.q_centers.unsqueeze(0).expand(B, self.K, self.n_q)  # [B,K,n_q]
        sig = torch.exp(self.q_logsigma).unsqueeze(0).expand(B, self.K, self.n_q) + 1e-3
        dist2 = ((q_exp - mu) / sig).pow(2).sum(dim=2)          # [B,K]
        w = torch.softmax(-0.5*dist2, dim=1)
        return w

    def _unpack_vertices(self, B):
        # produce stacked [K, L*L + L*N + L] from embed codes

        # Make sure indices live on the same device as the embedding weights
        device = self.vertex_embed.weight.device
        idx = torch.arange(self.K, device=device, dtype=torch.long)

        codes = self.vertex_embed(idx)  # [K, 32]
        raw = self.vert_mlp(codes)      # [K, out_dim]
        return raw

    def _mix_ABC(self, q):
        # returns A,B,c mixed by RBF weights
        Bsz = q.size(0)
        raw = self._unpack_vertices(Bsz)  # [K, out_dim]
        L, N, K = self.lat_space, self.n_control, self.K

        splitA = L*L
        splitB = splitA + L*N

        A_k = raw[:, :splitA].view(K, L, L)
        B_k = raw[:, splitA:splitB].view(K, L, N)
        c_k = raw[:, splitB:].view(K, L)

        w = self._rbf_weights(q)  # [B,K]
        # mix over K
        A = torch.einsum('bk,kij->bij', w, A_k)  # [B,L,L]
        Bm = torch.einsum('bk,kij->bij', w, B_k) # [B,L,N]
        c = torch.einsum('bk,ki->bi', w, c_k)    # [B,L]
        return A, Bm, c

    def forward(self, x, u, t):
        q = u[:, -self.n_q:] if self.n_q > 0 else torch.zeros(x.size(0), 1, device=x.device)
        A, Bm, c = self._mix_ABC(q)
        core = torch.einsum('bij,bj->bi', A, x) + torch.einsum('bij,bj->bi', Bm, u) + c
        rinp = torch.cat([x, u, t, q], dim=1)
        r = self.residual(rinp)
        return core + self.res_gain * r


# -----------------------------
#  LoRA-conditioned residual MLP
# -----------------------------
class _LoRALinear(nn.Module):
    """
    Standard LoRA adapter around a Linear layer: W + (alpha/r) * BA
    """
    def __init__(self, in_f, out_f, r=8, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Linear(in_f, r, bias=False)
            self.B = nn.Linear(r, out_f, bias=False)
        else:
            self.A = None
            self.B = None

    def forward(self, x):
        base = self.linear(x)
        if self.r > 0:
            return base + (self.alpha / self.r) * self.B(self.A(x))
        return base


class _FiLM(nn.Module):
    """
    FiLM modulation from q: for a hidden layer of size H,
        h <- gamma(q) * h + beta(q)
    """
    def __init__(self, n_q, H):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(n_q, H), nn.SiLU(), nn.Linear(H, H))
        self.beta  = nn.Sequential(nn.Linear(n_q, H), nn.SiLU(), nn.Linear(H, H))

    def forward(self, h, q):
        return self.gamma(q) * h + self.beta(q)


class _LoRAResidualMLP(nn.Module):
    """
    Residual MLP with LoRA layers and FiLM from q. Input: [x,u,t] (+ optionally q),
    Output: latent derivative.
    """
    def __init__(self, in_dim, out_dim, n_q, hsizes, r=8, alpha=1.0, include_q_in_input=True):
        super().__init__()
        self.include_q_in_input = include_q_in_input
        self.n_q = n_q

        dims = [in_dim] + hsizes + [out_dim]
        self.layers = nn.ModuleList([])
        self.films = nn.ModuleList([])

        for i in range(len(dims)-1):
            self.layers.append(_LoRALinear(dims[i], dims[i+1], r=r, alpha=alpha))
            if i < len(dims)-1:  # no FiLM on final layer output
                self.films.append(_FiLM(n_q, dims[i+1]))
        self.act = nn.SiLU()

    def forward(self, xin, q):
        # xin: [B, in_dim], q: [B, n_q]
        h = xin
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers)-1:
                h = self.act(self.films[i](h, q))
        return h


class LoRA_LPVResidual_NODE(nn.Module):
    """
    LPV core from q + LoRA+FiLM residual:
       f(x,u,t) = A(q)x + B(q)u + c(q) + r_LoRA([x,u,t], q)
    """
    def __init__(self, lat_space, n_control, n_layers, n_units, n_q=1,
                 device="cpu", r=8, alpha=1.0, res_hmult=0.5):
        super().__init__()
        self.lat_space = lat_space
        self.n_control = n_control
        self.in_features = lat_space + n_control
        self.out_features = lat_space
        self.n_q = n_q

        L, N = lat_space, n_control
        hid = max(32, min(128, 2*n_units))
        out_dim = L*L + L*N + L
        self.q2ABC = nn.Sequential(
            nn.Linear(n_q, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, out_dim)
        ).to(device)

        res_units = max(16, int(res_hmult*n_units))
        hsizes = [res_units]*n_layers
        self.residual = _LoRAResidualMLP(L + N + 1, L, n_q, hsizes, r=r, alpha=alpha).to(device)
        self.res_gain = nn.Parameter(torch.tensor(0.1)).to(device)

    def _split_ABC(self, vec):
        L, N = self.lat_space, self.n_control
        A_flat = vec[:, :L*L]
        B_flat = vec[:, L*L:L*L + L*N]
        c      = vec[:, L*L + L*N:]
        A = A_flat.view(-1, L, L)
        B = B_flat.view(-1, L, N)
        return A, B, c

    def forward(self, x, u, t):
        q = u[:, -self.n_q:] if self.n_q > 0 else torch.zeros(x.size(0), 1, device=x.device)
        abc = self.q2ABC(q)
        A, B, c = self._split_ABC(abc)
        core = torch.einsum('bij,bj->bi', A, x) + torch.einsum('bij,bj->bi', B, u) + c
        rinp = torch.cat([x, u, t], dim=1)
        r = self.residual(rinp, q)
        return core + self.res_gain * r

class SpectralSubmanifold_NODE(nn.Module):
    """
    Spectral-submanifold-style NODE in latent space.

    It keeps your q-conditioned LPV residual core (either LPVResidual or
    RBF_LPVResidual) and adds a low-dimensional spectral submanifold
    correction:

        x' = f_core(x, u, t; q) + g * (D W(z) z'),   z = E x,

    where W and z' are represented by MLPs and g is a trainable scalar gain.
    This way the SSM term cannot completely destroy the good behaviour
    of the LPV core.
    """
    def __init__(
        self,
        lat_space,
        n_control,
        n_layers,
        n_units,
        n_q=1,
        device="cpu",
        ssm_dim=None,
        core_type="rbf",
        K=6,
        res_hmult=0.5,
        ssm_gain_init=0.1,
    ):
        super().__init__()
        self.lat_space = lat_space
        self.n_control = n_control
        self.in_features = lat_space + n_control
        self.out_features = lat_space
        self.n_q = n_q
        self.device = device

        # Reduced spectral coordinates dimension
        if ssm_dim is None:
            ssm_dim = max(2, min(lat_space, lat_space // 2))
        self.ssm_dim = ssm_dim

        # Core LPV residual model (same family you already use)
        if core_type == "rbf":
            self.core = RBF_LPVResidual_NODE(
                lat_space=lat_space,
                n_control=n_control,
                n_layers=n_layers,
                n_units=n_units,
                n_q=n_q,
                K=K,
                device=device,
                res_hmult=res_hmult,
            )
        elif core_type == "lpv":
            self.core = LPVResidual_NODE(
                lat_space=lat_space,
                n_control=n_control,
                n_layers=n_layers,
                n_units=n_units,
                n_q=n_q,
                device=device,
                A_norm_cap=0.9,
                res_hmult=res_hmult,
            )
        else:
            raise ValueError(f"Unknown core_type '{core_type}' for SpectralSubmanifold_NODE.")

        # Linear encoder to spectral coordinates z = E x
        self.encoder_z = nn.Linear(lat_space, ssm_dim, bias=False).to(device)

        # Immersion W: z -> x (latent space)
        W_hsizes = [n_units] * n_layers
        self.W = blocks.MLP(
            ssm_dim,
            lat_space,
            bias=True,
            linear_map=nn.Linear,
            nonlin=nn.SiLU,
            hsizes=W_hsizes,
        ).to(device)

        # Reduced dynamics z' = R(z, u_phys, t, q)
        # we split u into [u_phys, q]
        R_in_dim = ssm_dim + (n_control - n_q) + 1 + n_q
        R_hsizes = [n_units] * n_layers
        self.R = blocks.MLP(
            R_in_dim,
            ssm_dim,
            bias=True,
            linear_map=nn.Linear,
            nonlin=nn.SiLU,
            hsizes=R_hsizes,
        ).to(device)

        # Global gain on the SSM correction
        self.ssm_gain = nn.Parameter(
            torch.tensor(float(ssm_gain_init), dtype=torch.float32)
        ).to(device)

    def forward(self, x, u, t):
        """
        x: [B, L]       latent state
        u: [B, n_u]     control + parameters q (last n_q entries)
        t: [B, 1]       normalized time

        Returns:
            x_dot: [B, L] latent derivative with LPV core + SSM correction.
        """
        # Core LPV residual field
        base = self.core(x, u, t)

        # If gain is (near) zero, skip the SSM branch for efficiency
        if torch.abs(self.ssm_gain) < 1e-6:
            return base

        # Split physical control vs parameters
        if self.n_q > 0:
            u_phys = u[:, :-self.n_q]
            q = u[:, -self.n_q:]
        else:
            u_phys = u
            q = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

        # Spectral coordinates
        z = self.encoder_z(x)  # [B, d]

        # Immersion
        def W_fn(z_):
            return self.W(z_)

        # Reduced dynamics on manifold
        R_inp = torch.cat([z, u_phys, t, q], dim=1)  # [B, d + n_u_phys + 1 + n_q]
        z_dot = self.R(R_inp)                        # [B, d]

        # x_ssm = D W(z) z_dot via forward-mode Jacobian-vector product
        x_ssm, _ = AF.jvp(W_fn, (z,), (z_dot,), create_graph=True)

        return base + self.ssm_gain * x_ssm



# -------------------------
#  EDM wrapper (unchanged)
# -------------------------
class EDM:
    def __init__(self, A_mat, pinv_Theta, dt, config, device='cpu'):
        self.A_mat = A_mat
        self.pinv_Theta = pinv_Theta
        self.dt = dt

        self.n_sparse = config['sensors']['n_sensors']

        self.Encoder_hsizes = config['model']['E_hsizes']
        self.Decoder_hsizes = config['model']['D_hsizes']

        self.n_NODE_layers = config['model']['n_layers']
        self.n_NODE_units  = config['model']['n_units']
        self.lat_space     = config['model']['lat_space']
        self.n_control     = config['model']['n_control']
        self.noise_std     = config['model']['noise']

        self.n_epoch     = config['training']['n_epoch']
        self.patience    = config['training']['patience']
        self.warmup      = config['training']['warmup']
        self.lr_patience = config['training']['lr_patience']
        self.lr          = config['training']['lr']
        self.Qs          = config['training']['Qs']

        self.lMB  = config['training']['lMB']
        self.nMB  = config['training']['nMB']
        self.nBPP = config['training']['nBPP']

        self.device = device
        self.problem = None

    def build_model(self, gate_config="regular", indx_q=1, method="rbf_lpv_resid"):
        """
        method ∈ {"lpv_resid", "rbf_lpv_resid", "lora_lpv_resid", "phase_norm", "softgate"}
        """
        Encoder_init = encoder(self.n_sparse, self.lat_space, self.Encoder_hsizes, self.device)
        Decoder_init = decoder(self.n_sparse, self.lat_space, self.Decoder_hsizes, self.device)
        fu = blocks.MLP(self.n_control, self.n_control, bias=True,
                        linear_map=torch.nn.Linear, nonlin=torch.nn.ELU,
                        hsizes=2*[60]).to(self.device)

        if method == "lpv_resid":
            NODE_init = LPVResidual_NODE(self.lat_space, self.n_control,
                                         self.n_NODE_layers, self.n_NODE_units,
                                         n_q=indx_q, device=self.device,
                                         A_norm_cap=0.9, res_hmult=0.5)
        elif method == "rbf_lpv_resid":
            NODE_init = RBF_LPVResidual_NODE(
                lat_space=self.lat_space,
                n_control=self.n_control,
                n_layers=self.n_NODE_layers,
                n_units=self.n_NODE_units,
                n_q=indx_q,
                K=6,
                device=self.device,
                res_hmult=0.5
            )
        elif method == "lora_lpv_resid":
            NODE_init = LoRA_LPVResidual_NODE(
                lat_space=self.lat_space,
                n_control=self.n_control,
                n_layers=self.n_NODE_layers,
                n_units=self.n_NODE_units,
                n_q=indx_q,
                device=self.device,
                r=8, alpha=1.0, res_hmult=0.5
            )
        elif method == "ssm":
            NODE_init = SpectralSubmanifold_NODE(
                lat_space=self.lat_space,
                n_control=self.n_control,
                n_layers=self.n_NODE_layers,
                n_units=self.n_NODE_units,
                n_q=indx_q,
                device=self.device,
                ssm_dim=None,      # or set explicitly if you want (e.g., 4)
                core_type="rbf",   # or "lpv" if you prefer that core
                K=6,
                res_hmult=0.5,
                ssm_gain_init=0.1,
            )
        elif method == "phase_norm":
            NODE_init = PhaseNorm_NODE(self.lat_space, self.n_control,
                                       self.n_NODE_layers, self.n_NODE_units,
                                       n_q=indx_q, gate_config=gate_config,
                                       device=self.device, sigma=0.08)
        else:
            NODE_init = softgate_NODE(self.lat_space, self.n_control,
                                      self.n_NODE_layers, self.n_NODE_units,
                                      self.dt, gate_config, device=self.device)

        fxRK4 = integrators.RK4(NODE_init, h=self.dt).to(self.device)
        noise_init = noiseLayer(std=self.noise_std, device=self.device)
        FS_init = full_space(self.n_sparse, self.A_mat, self.pinv_Theta, self.device)

        # Nodes
        encoder_x0 = Node(Encoder_init, ["x0"], ["LS_x0"], name="Encoder_x")
        noiseBlock = Node(noise_init, ['LS_x0'], ['LS_x'], name='Noise')
        fu_node    = Node(fu, ['U'], ['U'], name="Control_Encoder")
        model      = Node(fxRK4, ['LS_x', 'U', 't'], ['LS_x'], name='NODE')
        decoder_x  = Node(Decoder_init, ["LS_x"], ["x_hat"], name="Decoder_x")

        encoder_FX = Node(Encoder_init, ["X"], ["LS_X"], name="Encoder_X")
        decoder_FX = Node(Decoder_init, ["LS_X"], ["X_hat"], name="Decoder_X")

        dynamics_model = System([model], name='NODE_System', nsteps=self.lMB)

        FS_x = Node(FS_init, ["x_hat"], ["x_hat"], name="FS_x")
        FS_X = Node(FS_init, ["X_hat"], ["X_hat"], name="FS_X")
        FS_t = Node(FS_init, ["X"],     ["X"],     name="FS_t")

        # Losses
        X_true = variable("X")
        X_ae   = variable("X_hat")
        X_aen  = variable("x_hat")[:, :-1, :]

        ls_ae  = variable("LS_X")
        ls_aen = variable("LS_x")

        # temporal diff (finite difference along time)
        FDt_true = (X_true[:, 2:, :] - X_true[:, 1:-1, :])
        FDt_pred = (X_aen[:,  2:, :] - X_aen[:,  1:-1, :])

        # spatial diff (second central difference along spatial axis)
        CDx_true = (X_true[:, :, 3:] - 2*X_true[:, :, 2:-1] + X_true[:, :, 1:-2])
        CDx_pred = (X_aen[:,  :, 3:] - 2*X_aen[:,  :, 2:-1] + X_aen[:,  :, 1:-2])

        aenode_loss   = self.Qs["AENODE"]*(X_aen == X_true)^2;          aenode_loss.name   = "AENODE loss"
        ae_loss       = self.Qs["AE"]*(X_ae == X_true)^2;                ae_loss.name       = "AE loss"
        onestep_loss  = self.Qs["ONESTEP"]*(X_aen[:, 1, :] == X_true[:, 1, :])^2; onestep_loss.name  = "One Step loss"
        laststep_loss = self.Qs["LASTSTEP"]*(X_aen[:, -1, :] == X_true[:, -1, :])^2; laststep_loss.name = "Last Step loss"
        ls_loss       = self.Qs["LS"]*(ls_ae == ls_aen[:, :-1, :])^2;    ls_loss.name       = "Latent Space Loss"
        tdf_loss      = self.Qs["TEMPORALDIFF"]*(FDt_pred == FDt_true)^2; tdf_loss.name      = "Temporal Diff Loss"
        xdf_loss      = self.Qs["SPATIALDIFF"]*(CDx_pred == CDx_true)^2;  xdf_loss.name      = "Spatial Diff Loss"
        objectives = [aenode_loss, ae_loss, onestep_loss, laststep_loss, ls_loss, tdf_loss, xdf_loss]
        loss = PenaltyLoss(objectives, [])

        self.problem = Problem(
            [encoder_x0, encoder_FX, noiseBlock, fu_node, dynamics_model, decoder_x, decoder_FX, FS_x, FS_X, FS_t],
            loss
        )
        self.optimizer = torch.optim.Adam(self.problem.parameters(), lr=self.lr)
        self.problem.show()

    # ------------- data API (unchanged) -------------
    def get_data(self, X_train, U_train, X_dev, U_dev, X_test, U_test, t):
        nt = X_train.shape[1]
        nx = X_train.shape[-1]
        nu = U_train.shape[-1]

        # -------------- build TRAIN -----------------
        n_param = X_train.shape[0]
        s = np.random.choice(np.arange(nt, dtype=np.int64), [n_param, self.nBPP], replace=True)

        trainX = np.zeros([n_param, self.nBPP, self.lMB, nx])
        trainU = np.zeros([n_param, self.nBPP, self.lMB, nu])
        traint = np.zeros([n_param, self.nBPP, self.lMB, 1])

        for i in range(n_param):
            for j in range(self.nBPP):
                if s[i, j]+self.lMB < nt:
                    trainX[i, j] = X_train[i, s[i,j] : s[i,j]+self.lMB]
                    trainU[i, j] = U_train[i, s[i,j] : s[i,j]+self.lMB]
                    traint[i, j] = t[s[i,j] : s[i,j]+self.lMB]
                else:
                    temp1 = nt - s[i, j]
                    temp2 = self.lMB - temp1
                    trainX[i, j, :temp1] = X_train[i, s[i,j]:]
                    trainU[i, j, :temp1] = U_train[i, s[i,j]:]
                    traint[i, j, :temp1] = t[s[i,j]:]
                    trainX[i, j, temp1:] = X_train[i, :temp2]
                    trainU[i, j, temp1:] = U_train[i, :temp2]
                    traint[i, j, temp1:] = t[:temp2]

        # --------------- build DEV -------------------
        n_param = X_dev.shape[0]
        s = np.random.choice(np.arange(nt, dtype=np.int64), [n_param, self.nBPP], replace=True)

        devX = np.zeros([n_param, self.nBPP, self.lMB, nx])
        devU = np.zeros([n_param, self.nBPP, self.lMB, nu])
        devt = np.zeros([n_param, self.nBPP, self.lMB, 1])

        for i in range(n_param):
            for j in range(self.nBPP):
                if s[i, j]+self.lMB < nt:
                    devX[i, j] = X_dev[i, s[i,j] : s[i,j]+self.lMB]
                    devU[i, j] = U_dev[i, s[i,j] : s[i,j]+self.lMB]
                    devt[i, j] = t[s[i,j] : s[i,j]+self.lMB]
                else:
                    temp1 = nt - s[i, j]
                    temp2 = self.lMB - temp1
                    devX[i, j, :temp1] = X_dev[i, s[i,j]:]
                    devU[i, j, :temp1] = U_dev[i, s[i,j]:]
                    devt[i, j, :temp1] = t[s[i,j]:]
                    devX[i, j, temp1:] = X_dev[i, :temp2]
                    devU[i, j, temp1:] = U_dev[i, :temp2]
                    devt[i, j, temp1:] = t[:temp2]

        # torchify + DataLoaders
        trainX = torch.tensor(np.concatenate(trainX, axis=0), dtype=torch.float32, device=self.device)
        trainU = torch.tensor(np.concatenate(trainU, axis=0), dtype=torch.float32, device=self.device)
        traint = torch.tensor(np.concatenate(traint, axis=0), dtype=torch.float32, device=self.device)
        train_data = DictDataset({'X': trainX, 'x0': trainX[:, 0:1, :], 'U': trainU, 't': traint}, name='train')
        train_loader = DataLoader(train_data, batch_size=self.nMB, collate_fn=train_data.collate_fn, shuffle=True)

        devX = torch.tensor(np.concatenate(devX, axis=0), dtype=torch.float32, device=self.device)
        devU = torch.tensor(np.concatenate(devU, axis=0), dtype=torch.float32, device=self.device)
        devt = torch.tensor(np.concatenate(devt, axis=0), dtype=torch.float32, device=self.device)
        dev_data = DictDataset({'X': devX, 'x0': devX[:, 0:1, :], 'U': devU, 't': devt}, name='val')
        dev_loader = DataLoader(dev_data, batch_size=self.nMB, collate_fn=dev_data.collate_fn, shuffle=False)

        # --------------- TEST (full sequences) ---------------
        testX = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        testU = torch.tensor(U_test, dtype=torch.float32, device=self.device)
        testt = torch.tensor(t, dtype=torch.float32, device=self.device)
        test_data = {'X': testX, 'x0': testX[:, 0:1, :], 'U': testU, 't': testt.repeat(X_test.shape[0], 1, 1)}

        return train_loader, dev_loader, test_data

    def train_model(self, output_paths, data_train, ft_train, data_dev, ft_dev, data_test, ft_test, t):
        train_loader, dev_loader, test_data = self.get_data(
            data_train, ft_train, data_dev, ft_dev, data_test, ft_test, t
        )

        callbacker = custom_callback(self.device)
        logger = BasicLogger(args=None, savedir=output_paths, verbosity=1, stdout=['val_loss', 'train_loss'])

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
        )

        best_model = trainer.train()
        self.problem.load_state_dict(best_model)
        return self.problem
