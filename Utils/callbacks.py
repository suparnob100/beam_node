# Modified Callbacks

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

    
class Callback:
    """
    Callback base class which allows for bare functionality of Trainer
    """
    def __init__(self, device):
        self.device = device
        self.current_lr = 999.0

    def begin_train(self, trainer):
        self.current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"\n Current Learning Rate - {self.current_lr}")
        
    def begin_epoch(self, trainer, output):
        pass

    def begin_eval(self, trainer, output):
        pass

    def end_batch(self, trainer, output):
        pass

    def end_eval(self, trainer, output):
        pass

    def end_epoch(self, trainer, output):
        temp = trainer.optimizer.param_groups[0]['lr']
        if self.current_lr != temp:
            self.current_lr = temp
            print(f"\n New Learning Rate - {temp}")
            
        if trainer.current_epoch%10 == 0 and False:
            t_test = t_test = torch.linspace(0, 499, 500).to(trainer.device)
            t_test = t_test[:, torch.newaxis]/500
            raw_weights = trainer.model.nodes[1].nodes[0].callable.block.weight_net(t_test)
            weights = nn.functional.softmax(raw_weights, dim=-1).unsqueeze(1)
    
            weights = weights.detach().cpu().numpy()
            
            plt.figure()
            plt.plot(weights[:, 0, :])
            plt.show()
            
        if trainer.current_epoch%10 == 0 and False:
            t_test = t_test = torch.linspace(0, 499, 500).to(trainer.device)
            t_test = t_test[:, torch.newaxis]/500
            raw_weights = trainer.model.nodes[1].nodes[0].callable.block.gaussian_weights(t_test)
    
            weights = raw_weights.detach().cpu().numpy()
            
            plt.figure()
            plt.plot(weights[:, 0, :])
            plt.show()

    def end_train(self, trainer, output):
        pass

    def begin_test(self, trainer):
        pass

    def end_test(self, trainer, output):
        pass