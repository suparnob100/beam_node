# Modified version of Neuromancer Trainer
from copy import deepcopy
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neuromancer.loggers import BasicLogger
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback
from neuromancer.dataset import DictDataset



def move_batch_to_device(batch, device="cpu"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

class custom_callback(Callback):
    def __init__(self, device='cpu'):
        self.device = device
        self.current_lr = 999.0

    def begin_train(self, trainer):
        self.current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"\n Current Learning Rate - {self.current_lr}")

    def end_epoch(self, trainer, output):
        temp = trainer.optimizer.param_groups[0]['lr']
        if self.current_lr != temp:
            self.current_lr = temp
            print(f"\n New Learning Rate - {temp}")

class Trainer:

    def __init__(
        self,
        problem: Problem,
        train_data: torch.utils.data.DataLoader,
        dev_data: torch.utils.data.DataLoader = None,
        test_data: torch.utils.data.DataLoader = None,
        optimizer: torch.optim.Optimizer = None,
        logger: BasicLogger = None,
        callback: Callback = custom_callback,
        lr_scheduler=False,
        epochs=1000,
        epoch_verbose=1,
        patience=5,
        warmup=0,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        eval_mode="min",
        clip=100.0,
        device="cpu"
    ):

        self.model = problem
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(problem.parameters(), 0.01, betas=(0.0, 0.9))
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
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.25, patience=lr_scheduler, verbose=True)
        self.patience = patience
        self.warmup = warmup
        self.badcount = 0
        self.clip = clip
        self.best_devloss = np.finfo(np.float32).max if self._eval_min else 0.
        self.best_model = deepcopy(self.model.state_dict())
        self.device = device

    def train(self, trial=None):

        self.callback.begin_train(self)

        try:
            for i in range(self.current_epoch, self.current_epoch+self.epochs):

                self.model.train()
                losses = []
                for t_batch in self.train_data:
                    t_batch['epoch'] = i
                    t_batch = move_batch_to_device(t_batch, self.device)
                    output = self.model(t_batch)
                    self.optimizer.zero_grad()
                    output[self.train_metric].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    losses.append(output[self.train_metric])
                    self.callback.end_batch(self, output)

                output[f'mean_{self.train_metric}'] = torch.mean(torch.stack(losses))
                self.callback.begin_epoch(self, output)

                

                with torch.set_grad_enabled(self.model.grad_inference):
                    self.model.eval()
                    if self.dev_data is not None:
                        losses = []
                        for d_batch in self.dev_data:
                            d_batch = move_batch_to_device(d_batch, self.device)
                            eval_output = self.model(d_batch)
                            for key, value in d_batch.items():
                                if isinstance(value, torch.Tensor):
                                    batch_size = value.shape[0]
                                    break
                            losses.append(eval_output[self.dev_metric]/batch_size)
                        eval_output[f'mean_{self.dev_metric}'] = torch.mean(torch.stack(losses))
                        output = {**output, **eval_output}
                    self.callback.begin_eval(self, output)

                    if (self._eval_min and output[self.eval_metric] < self.best_devloss)\
                            or (not self._eval_min and output[self.eval_metric] > self.best_devloss):
                        self.best_model = deepcopy(self.model.state_dict())
                        self.best_devloss = output[self.eval_metric]
                        self.badcount = 0
                    else:
                        if i > self.warmup:
                            self.badcount += 1
                    if self.logger is not None:
                        self.logger.log_metrics(output, step=i)
                    else:
                        mean_loss = output[f'mean_{self.train_metric}']
                        if i % (self.epoch_verbose) == 0:
                            print(f'epoch: {i}  {self.train_metric}: {mean_loss}')

                    self.callback.end_eval(self, output)  # visualizations

                    self.callback.end_epoch(self, output)

                    if self.badcount > self.patience:
                        print('Early stopping!!!')
                        break
                    self.current_epoch = i + 1
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(output[self.eval_metric])

        except KeyboardInterrupt:
            print("Interrupted training loop.")

        self.callback.end_train(self, output)  # write training visualizations

        # Assign best weights to the model
        self.model.load_state_dict(self.best_model)

        if self.logger is not None:
            self.logger.log_artifacts({
                "best_model_state_dict.pth": self.best_model,
                "best_model.pth": self.model,
            })
        return self.best_model