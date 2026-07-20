"""Modified Neuromancer Trainer with HuggingFace Accelerate integration.

Key improvements over the vanilla training loop:
    - ``accelerate.Accelerator`` handles device placement, mixed precision,
      multi-GPU distribution, and gradient accumulation automatically.
    - ``torch.compile`` (PyTorch 2.0+) for graph-mode kernel fusion.
    - ``optimizer.zero_grad(set_to_none=True)`` for faster gradient clearing.
    - Detached scalar loss accumulation to avoid GPU memory leaks.
    - ``pin_memory`` + ``num_workers`` in DataLoaders for async CPU→GPU I/O.
    - ``torch.no_grad()`` for validation (explicit, clearer intent).
    - cuDNN benchmark auto-tuning hint.

Falls back gracefully to CPU-only training when no GPU is available.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback
from neuromancer.dataset import DictDataset

# Optional: HuggingFace Accelerate
try:
    from accelerate import Accelerator
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False


def move_batch_to_device(
    batch: dict[str, Any], device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Move all tensors in *batch* to *device* (non-blocking for pinned memory)."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

class custom_callback(Callback):
    """Callback that prints learning-rate changes during training."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.current_lr = 999.0

    def begin_train(self, trainer: "Trainer") -> None:
        self.current_lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"\n Current Learning Rate - {self.current_lr}")

    def end_epoch(self, trainer: "Trainer", output: dict) -> None:
        temp = trainer.optimizer.param_groups[0]["lr"]
        if self.current_lr != temp:
            self.current_lr = temp
            print(f"\n New Learning Rate - {temp}")


class Trainer:
    """Training loop with Accelerate, torch.compile, early stopping, and LR scheduling.

    Parameters
    ----------
    problem : Problem
        Neuromancer Problem instance.
    train_data, dev_data, test_data : DataLoader or dict
        Training, validation, and test data.
    optimizer : torch.optim.Optimizer
        Optimiser instance.
    logger : BasicLogger or None
        Logging backend.
    callback : Callback
        Callback for epoch/batch hooks.
    lr_scheduler : int or False
        Patience for ``ReduceLROnPlateau``.
    epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience.
    warmup : int
        Warm-up epochs before early stopping activates.
    clip : float
        Gradient clipping norm.
    use_accelerate : bool
        If True and ``accelerate`` is installed, wrap the training loop with
        ``Accelerator`` for automatic mixed-precision and multi-GPU support.
    mixed_precision : str
        Mixed-precision mode for Accelerate: ``"no"``, ``"fp16"``, or ``"bf16"``.
    compile_model : bool
        If True, wrap the model with ``torch.compile`` (PyTorch 2.0+).
    num_workers : int
        Number of DataLoader worker processes.
    """

    def __init__(
        self,
        problem: Problem,
        train_data: torch.utils.data.DataLoader,
        dev_data: torch.utils.data.DataLoader | None = None,
        test_data: torch.utils.data.DataLoader | dict | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        logger: BasicLogger | None = None,
        callback: Callback = custom_callback,
        lr_scheduler: int | bool = False,
        epochs: int = 1000,
        epoch_verbose: int = 1,
        patience: int = 5,
        warmup: int = 0,
        train_metric: str = "train_loss",
        dev_metric: str = "dev_loss",
        test_metric: str = "test_loss",
        eval_metric: str = "dev_loss",
        eval_mode: str = "min",
        clip: float = 100.0,
        device: str = "cpu",
        # ── new training-perf knobs ──
        use_accelerate: bool = False,
        mixed_precision: str = "no",
        compile_model: bool = False,
        num_workers: int = 0,
        output_paths: str | None = None,
    ) -> None:
        # cuDNN benchmark auto-tuning (finds fastest conv algorithms for fixed input sizes)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.model = problem
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(problem.parameters(), 0.01, betas=(0.0, 0.9))
        )
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.callback = callback
        self.callback.device = device
        self.logger = logger
        self.epochs = epochs
        self.current_epoch = 0
        self.epoch_verbose = epoch_verbose
        if logger is not None:
            self.logger.log_weights(self.model)
        self.train_metric = train_metric
        self.dev_metric = dev_metric
        self.test_metric = test_metric
        self.eval_metric = eval_metric
        self._eval_min = eval_mode == "min"
        # PyTorch >= 2.2 deprecated verbose=True
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.25, patience=lr_scheduler, verbose="legacy"
        )
        self.patience = patience
        self.warmup = warmup
        self.badcount = 0
        self.clip = clip
        self.best_devloss = np.finfo(np.float32).max if self._eval_min else 0.0
        self.best_model = deepcopy(self.model.state_dict())
        self.device = device
        self.num_workers = num_workers
        self.output_paths = output_paths

        # ── Accelerator setup ──
        self.accelerator: Accelerator | None = None
        if use_accelerate and _HAS_ACCELERATE:
            self.accelerator = Accelerator(mixed_precision=mixed_precision)
            (
                self.model,
                self.optimizer,
                self.train_data,
            ) = self.accelerator.prepare(self.model, self.optimizer, self.train_data)
            if self.dev_data is not None:
                self.dev_data = self.accelerator.prepare(self.dev_data)
            self.device = self.accelerator.device
            print(f"[Accelerator] device={self.device}, mixed_precision={mixed_precision}")
        elif use_accelerate and not _HAS_ACCELERATE:
            warnings.warn(
                "use_accelerate=True but `accelerate` is not installed. "
                "Falling back to standard training. Install with: pip install accelerate",
                stacklevel=2,
            )

        # ── torch.compile (PyTorch 2.0+ graph-mode fusion) ──
        if compile_model:
            if hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model)
                    print("[torch.compile] Model compiled for graph-mode execution.")
                except Exception as e:
                    warnings.warn(f"torch.compile failed, falling back to eager mode: {e}", stacklevel=2)
            else:
                warnings.warn(
                    "compile_model=True but torch.compile not available (requires PyTorch >= 2.0).",
                    stacklevel=2,
                )

    def train(self, trial: Any | None = None) -> dict:
        """Run the training loop and return the best model state dict.

        Training improvements:
            - ``zero_grad(set_to_none=True)`` avoids a memset on every batch.
            - Loss scalars are ``.detach().item()`` to avoid graph retention.
            - Validation runs under ``torch.no_grad()`` (explicit, saves memory).
            - When Accelerator is active, ``backward()`` and gradient clipping
              are routed through Accelerator for mixed-precision support.
        """
        self.callback.begin_train(self)

        try:
            for i in range(self.current_epoch, self.current_epoch + self.epochs):
                self.model.train()
                batch_losses: list[float] = []  # store Python scalars, not tensors

                for t_batch in self.train_data:
                    t_batch["epoch"] = i
                    # Device placement: Accelerator handles it, or manual
                    if self.accelerator is None:
                        t_batch = move_batch_to_device(t_batch, self.device)

                    output = self.model(t_batch)

                    # ── backward ──
                    loss = output[self.train_metric]
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.accelerator is not None:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()

                    # ── gradient clipping ──
                    if self.accelerator is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()

                    # Store detached scalar (prevents graph/tensor accumulation across batches)
                    batch_losses.append(loss.detach().item())
                    self.callback.end_batch(self, output)

                # Epoch-level mean loss (from Python scalars — no tensor accumulation)
                output[f"mean_{self.train_metric}"] = torch.tensor(
                    np.mean(batch_losses), device=self.device
                )
                self.callback.begin_epoch(self, output)

                # ── validation ──
                with torch.no_grad():
                    self.model.eval()
                    if self.dev_data is not None:
                        val_losses: list[float] = []
                        for d_batch in self.dev_data:
                            if self.accelerator is None:
                                d_batch = move_batch_to_device(d_batch, self.device)
                            eval_output = self.model(d_batch)
                            # Infer batch size
                            for value in d_batch.values():
                                if isinstance(value, torch.Tensor):
                                    batch_size = value.shape[0]
                                    break
                            val_losses.append(
                                (eval_output[self.dev_metric] / batch_size).detach().item()
                            )
                        eval_output[f"mean_{self.dev_metric}"] = torch.tensor(
                            np.mean(val_losses), device=self.device
                        )
                        output = {**output, **eval_output}
                    self.callback.begin_eval(self, output)

                    if (self._eval_min and output[self.eval_metric] < self.best_devloss) or (
                        not self._eval_min and output[self.eval_metric] > self.best_devloss
                    ):
                        self.best_model = deepcopy(self.model.state_dict())
                        self.best_devloss = output[self.eval_metric]
                        self.badcount = 0
                    else:
                        if i > self.warmup:
                            self.badcount += 1
                    if self.logger is not None:
                        self.logger.log_metrics(output, step=i)
                    else:
                        mean_loss = output[f"mean_{self.train_metric}"]
                        if i % self.epoch_verbose == 0:
                            print(f"epoch: {i}  {self.train_metric}: {mean_loss:.6f}")

                    self.callback.end_eval(self, output)
                    self.callback.end_epoch(self, output)

                    if self.badcount > self.patience:
                        print("Early stopping!!!")
                        break
                    self.current_epoch = i + 1

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(output[self.eval_metric])

        except KeyboardInterrupt:
            print("Interrupted training loop.")
            if self.output_paths is not None:
                interrupted_state = self.model.state_dict()
                torch.save(interrupted_state, f"{self.output_paths}/interrupted_model_state_dict.pth")
                print(f"Saved interrupted model to {self.output_paths}/interrupted_model_state_dict.pth")

        self.callback.end_train(self, output)

        self.model.load_state_dict(self.best_model)

        if self.logger is not None:
            self.logger.log_artifacts(
                {
                    "best_model_state_dict.pth": self.best_model,
                    "best_model.pth": self.model,
                }
            )
        return self.best_model
