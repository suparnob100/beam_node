"""EDM.py — shared building blocks for the EDM pipeline.

Provides the config-parsing base ``EDM`` class plus the reusable graph
components (encoder/decoder, full-space lift, noise layer, checkpointed RK4,
resampling callback) consumed by ``EDM_v1_1`` and ``EDM_v1_2``.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import functional as AF
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from neuromancer.dataset import DictDataset
from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.problem import Problem

# Local trainer — sys.path hack required because this package is not yet
# pip-installable and notebooks run from varying working directories.
if "__file__" in globals():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    _script_dir = os.getcwd()
_utils_dir = os.path.abspath(os.path.join(_script_dir, "..", "Utils"))
sys.path.append(_utils_dir)
from trainer import Trainer, custom_callback


class ResamplingCallback(custom_callback):
    """Rebuild the train DataLoader with fresh sliding windows every N epochs.

    Why: `_build_sliding_windows` random-samples `nBPP` window starts per
    parameter.  With a static dataset, a small `nBPP` permanently starves
    the NODE of trajectory regions.  Re-sampling every few epochs accumulates
    full coverage across the run while keeping `nBPP` (and batches/epoch)
    small for fast training.
    """

    def __init__(self, edm: "EDM", device: str = "cpu") -> None:
        super().__init__(device)
        self.edm = edm

    def end_epoch(self, trainer: "Trainer", output: dict) -> None:  # type: ignore[override]
        super().end_epoch(trainer, output)
        completed_epochs = trainer.current_epoch + 1
        if completed_epochs % self.edm.resample_every_epochs != 0:
            return
        new_loader = self.edm._build_train_loader()
        print(f"\n[ResamplingCallback] Rebuilt train DataLoader with {len(new_loader.dataset)} samples.")
        if trainer.accelerator is not None:
            new_loader = trainer.accelerator.prepare(new_loader)
        trainer.train_data = new_loader


# ───────────────────────────────────────────────────────────────────────
#  Shared building blocks
# ───────────────────────────────────────────────────────────────────────

class NoiseLayer(nn.Module):
    """Additive Gaussian noise during training (regularisation)."""

    def __init__(self, std: float = 0.005, device: str = "cpu") -> None:
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0.0:
            noise = torch.normal(0, self.std, size=x.size(), device=self.device)
            x = x + noise
        return x


class encoder(nn.Module):
    """Encode full-state RS = [displacements, velocities] (size ``2*n_sparse``)
    into a latent vector (size ``lat_space``)."""

    def __init__(
        self,
        n_sparse: int,
        lat_space: int,
        E_hsizes: list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.block = blocks.MLP(
            2 * n_sparse, lat_space, bias=True,
            linear_map=nn.Linear, nonlin=nn.SiLU,
            hsizes=E_hsizes,
        ).to(device)
        self.lin_layer = nn.Linear(lat_space, lat_space, bias=True).to(device)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(p=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.drop(self.lin_layer(out))
        return self.act(out)


class decoder(nn.Module):
    """Decode latent vector to RS (linear output head, no tanh)."""

    def __init__(
        self,
        n_sparse: int,
        lat_space: int,
        D_hsizes: list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.block = blocks.MLP(
            lat_space, 2 * n_sparse, bias=True,
            linear_map=nn.Linear, nonlin=nn.SiLU,
            hsizes=D_hsizes,
        ).to(device)
        self.lin_layer = nn.Linear(2 * n_sparse, 2 * n_sparse, bias=True).to(device)
        self.drop = nn.Dropout(p=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.drop(self.lin_layer(out))
        return out


class full_space(nn.Module):
    """Map from RS (sparse) to full space FS via ``C = (A @ pinv_Theta)^T``."""

    def __init__(
        self,
        n_sparse: int,
        A_Mat: NDArray,
        pinv_Theta: NDArray,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.C = (
            torch.tensor(A_Mat @ pinv_Theta, dtype=torch.float32, device=self.device)
            .T
            .unsqueeze(0)
        )
        self.n_sparse = n_sparse

    def forward(self, RS: torch.Tensor) -> torch.Tensor:
        RS1 = RS[:, :, : self.n_sparse]
        RS2 = RS[:, :, self.n_sparse :]
        FS1 = RS1 @ self.C
        FS2 = RS2 @ self.C
        return torch.cat([FS1, FS2], dim=-1)


# ───────────────────────────────────────────────────────────────────────
#  Gradient-checkpointed RK4 integrator
# ───────────────────────────────────────────────────────────────────────

class CheckpointedRK4(nn.Module):
    """Drop-in replacement for ``neuromancer.dynamics.integrators.RK4``
    with activation checkpointing on every RK4 stage.

    Standard RK4 evaluates the dynamics ``block`` 4 times per step and
    stores all intermediate activations for backprop.  For long rollouts
    this dominates GPU memory.  Checkpointing trades ~2× compute for
    ~4× activation-memory savings — often a net win because it enables
    larger batch sizes.

    The interface mirrors neuromancer's ``Integrator`` / ``RK4``:
    ``forward(x, *args) -> x_next``.
    """

    def __init__(self, block: nn.Module, h: float = 1.0) -> None:
        super().__init__()
        self.block = block
        self.in_features = block.in_features
        self.out_features = block.out_features
        self.h = h

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        h = self.h

        # Each k-stage is checkpointed: activations are recomputed during
        # the backward pass instead of being stored.
        # use_reentrant=False is the recommended PyTorch 2.x mode.
        def _block_fn(*inputs: torch.Tensor) -> torch.Tensor:
            return self.block(*inputs)

        k1 = _grad_checkpoint(_block_fn, x, *args, use_reentrant=False)
        k2 = _grad_checkpoint(_block_fn, x + h * k1 / 2.0, *args, use_reentrant=False)
        k3 = _grad_checkpoint(_block_fn, x + h * k2 / 2.0, *args, use_reentrant=False)
        k4 = _grad_checkpoint(_block_fn, x + h * k3, *args, use_reentrant=False)

        return x + h * (k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0)


class EDM:
    """Encoder-Decoder Model wrapper.

    Orchestrates model construction, data preparation, and training.
    """

    _CHECKPOINT_FILES = (
        "best_model_state_dict.pth",
        "interrupted_model_state_dict.pth",
    )
    _STATE_GROUP_NODE_NAMES = {
        "AE": ("Encoder_x", "Encoder_X", "Decoder_x", "Decoder_X"),
        "NODE": ("Control_Encoder", "NODE_System"),
    }

    def __init__(
        self,
        A_mat: NDArray,
        pinv_Theta: NDArray,
        dt: float,
        t_max: float,
        config: dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.t_max = t_max

        self.A_mat = A_mat
        self.pinv_Theta = pinv_Theta
        self.dt = dt

        self.n_sparse = config["sensors"]["n_sensors"]

        self.Encoder_hsizes = config["model"]["E_hsizes"]
        self.Decoder_hsizes = config["model"]["D_hsizes"]

        self.n_NODE_layers = config["model"]["n_layers"]
        self.n_NODE_units = config["model"]["n_units"]
        self.lat_space = config["model"]["lat_space"]
        self.n_control = config["model"]["n_control"]
        self.noise_std = config["model"]["noise"]

        self.n_epoch = config["training"]["n_epoch"]
        self.patience = config["training"]["patience"]
        self.warmup = config["training"]["warmup"]
        self.lr_patience = config["training"]["lr_patience"]
        self.lr = config["training"]["lr"]
        self.Qs = config["training"]["Qs"]

        self.lMB = config["training"]["lMB"]
        self.nMB = config["training"]["nMB"]
        self.nBPP = config["training"]["nBPP"]
        self.resample_every_epochs = max(
            1, int(config["training"].get("resample_every_epochs", 1))
        )

        self.method = None

        self.device = device
        self.problem: Problem | None = None
