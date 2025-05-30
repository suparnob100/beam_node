#%% Initialize
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import re
import glob


from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem


class noiseLayer(nn.Module):
    def __init__(self, std=0.005, device='cpu'):
        super(noiseLayer, self).__init__()
        self.std = std
        self.device = device
    def forward(self, x):
        if self.training:
            noise = torch.normal(0, self.std, size=x.size()).to(self.device)
            x = x + noise
        return x
    
class encoder(nn.Module):
    def __init__(self, n_sparse, lat_space, E_hsizes, device='cpu'):
        super(encoder, self).__init__()
        self.block = blocks.MLP(2*n_sparse, n_sparse, bias = True,
                        linear_map = torch.nn.Linear,
                        nonlin = torch.nn.SiLU,
                        hsizes=E_hsizes).to(device)
        self.lin_layer = nn.Linear(lat_space, lat_space, bias=True).to(device)
        self.tan = nn.Tanh()
        
    def forward(self, x):
        output = self.block(x)
        output = self.lin_layer(output)
        return self.tan(output)
        
class decoder(nn.Module):
    def __init__(self, n_sparse, lat_space, D_hsizes, device='cpu'):
        super(decoder, self).__init__()
        self.block = blocks.MLP(lat_space, 2*n_sparse, bias = True,
                        linear_map = torch.nn.Linear,
                        nonlin = torch.nn.SiLU,
                        hsizes=D_hsizes).to(device)
        self.lin_layer = nn.Linear(2*n_sparse, 2*n_sparse, bias=True).to(device)
        self.tan = nn.Tanh()
    
    def forward(self, x):
        output = self.block(x)
        output = self.lin_layer(output)
        return self.tan(output)

def NODE(lat_space, n_control, n_layers, n_units, dt, device='cpu'):
    fx = blocks.MLP(lat_space+n_control, lat_space, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=n_layers*[n_units]).to(device)

    fxRK4 = integrators.RK4(fx, h=dt)

    return fxRK4

class EDM:
    def __init__(self, config, device='cpu'):
        self.n_sparse = TODO

        self.Encoder_hsizes = config['model']['E_hsizes']
        self.Decoder_hsizes = config['model']['D_hsizes']

        self.n_NODE_layers = config['model']['n_layers']
        self.n_NODE_units = config['model']['n_units']
        
        self.lat_space = config['model']['lat_space']
        self.n_control = config['model']['n_control']
        self.noise_std = config['model']['noise']

        self.n_epoch = config['training']['n_epoch']
        self.patience = config['training']['patience']
        self.Qs = config['training']['Qs']

        self.device=device


    def build_model(self):
        Encoder_init = encoder()
        Decoder_init = decoder()
        NODE_init = NODE(self.lat_space, self.n_control, self.n_NODE_layers, self.n_NODE_units, device=self.device)
        noise_init = noiseLayer(std=self.noise_std)

        encoder_x0 = Node(Encoder_init, ["x0"], ["LS_x0"], name="Encoder_x")
        noiseBlock = Node(noise_init, ['LS_x0'], ['LS_x'], name='Noise')
        model = Node(NODE_init, ['LS_x', 'U'], ['LS_x'], name='NODE')
        decoder_x = Node(Decoder_init, [f"LS_x"], [f"x_hat"], name=f"Decoder_x")
        
        encoder_FX = Node(Encoder_init, [f"X"], [f"LS_X"], name=f"Encoder_X")
        decoder_FX = Node(Decoder_init, [f"LS_X"], [f"X_hat"], name=f"Decoder_X")

        dynamics_model = System([model], name='NODE_System', nsteps=CONFIG['lMB'])

    def train_model(self):
        pass