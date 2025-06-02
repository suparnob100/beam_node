#%% Initialize
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuromancer.dataset import DictDataset
from neuromancer.modules import blocks
from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger


if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

utils_dir = os.path.abspath(os.path.join(script_dir, "..", "Utils"))


from trainer import Trainer
from callbacks import Callback

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
        self.block = blocks.MLP(2*n_sparse, lat_space, bias = True,
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
    def __init__(self, A_mat, pinv_Theta, config, device='cpu'):
        self.A_mat = A_mat
        self.pinv_Theta = pinv_Theta

        self.n_sparse = config['sensors']['n_sensors']

        self.Encoder_hsizes = config['model']['E_hsizes']
        self.Decoder_hsizes = config['model']['D_hsizes']

        self.n_NODE_layers = config['model']['n_layers']
        self.n_NODE_units = config['model']['n_units']
        
        self.lat_space = config['model']['lat_space']
        self.n_control = config['model']['n_control']
        self.noise_std = config['model']['noise']

        self.n_epoch = config['training']['n_epoch']
        self.patience = config['training']['patience']
        self.warmup = config['training']['warmup']
        self.lr_patience = config['training']['lr_patience']
        self.lr = config['training']['lr']
        self.Qs = config['training']['Qs']

        self.lMB = config['training']['lMB']
        self.nMB = config['training']['nMB']
        self.nBPP = config['training']['nBPP']

        self.device = device

        self.problem = None


    def build_model(self):
        Encoder_init = encoder(self.n_sparse, self.lat_space, self.Encoder_hsizes, self.device)
        Decoder_init = decoder(self.n_sparse, self.lat_space, self.Decoder_hsizes, self.device)
        NODE_init = NODE(self.lat_space, self.n_control, self.n_NODE_layers, self.n_NODE_units, device=self.device)
        noise_init = noiseLayer(std=self.noise_std)

        encoder_x0 = Node(Encoder_init, ["x0"], ["LS_x0"], name="Encoder_x")
        noiseBlock = Node(noise_init, ['LS_x0'], ['LS_x'], name='Noise')
        model = Node(NODE_init, ['LS_x', 'U'], ['LS_x'], name='NODE')
        decoder_x = Node(Decoder_init, [f"LS_x"], [f"x_hat"], name=f"Decoder_x")
        
        encoder_FX = Node(Encoder_init, [f"X"], [f"LS_X"], name=f"Encoder_X")
        decoder_FX = Node(Decoder_init, [f"LS_X"], [f"X_hat"], name=f"Decoder_X")

        dynamics_model = System([model], name='NODE_System', nsteps=self.lMB)

        ## Loss functions
        # Variables
        x_true = variable("X")
        x_ae = variable("X_hat")
        x_aenode = variable("x_hat")
        
        ls_ae = variable("LS_X")
        ls_aenode = variable("LS_x")

        # Full Space Conversion
        C = torch.tensor(self.A_mat@self.pinv_Theta, dtype=torch.float32, device=self.device).T.unsqueeze(0) #Full space constant
        X_true = x_true@C
        X_ae = x_ae@C
        X_aenode = x_aenode[:, :-1, :]@C

        # Temporal differencing
        FDt_true = (X_true[:, 2:, :] - X_true[:, 1:-1, :])
        FDt_pred = (X_aenode[:, 2:, :] - X_aenode[:, 1:-1, :])

        # Spatial differencing
        CDx_true = (X_true[:, :, 3:] - 2*X_true[:, :, 2:-1] + X_true[:, :, 1:-2])
        CDx_pred = (X_aenode[:, :, 3:] - 2*X_aenode[:, :, 2:-1] + X_aenode[:, :, 1:-2])

        # AE+NODE Loss
        aenode_loss = self.Qs["AENODE"]*(X_aenode == X_true)^2
        aenode_loss.name = "AENODE loss"

        # AE Loss
        ae_loss = self.Qs["AE"]*(X_ae == X_true)^2
        ae_loss.name = "AE loss"

        # One Step Tracking Loss
        onestep_loss = self.Qs["ONESTEP"]*(X_aenode[:, 1, :] == X_true[:, 1, :])^2
        onestep_loss.name = "One Step loss"

        # Last Step Tracking Loss
        laststep_loss = self.Qs["LASTSTEP"]*(X_aenode[:, -1, :] == X_true[:, -1, :])^2
        laststep_loss.name = "Last Step loss"

        # Latent Space Loss
        ls_loss = self.Qs["LS"]*(ls_ae == ls_aenode[:, :-1, :])^2
        ls_loss.name = "Latent Space Loss"

        # Temporal Difference Loss
        tdf_loss = self.Qs["TEMPORALDIFF"]*(FDt_pred == FDt_true)^2
        tdf_loss.name = "Temporal Diff Loss"

        # Spatial Difference Loss
        xdf_loss = self.Qs["SPATIALDIFF"]*(CDx_pred == CDx_true)^2
        xdf_loss.name = "Spatial Diff Loss"


        objectives = [aenode_loss, ae_loss, onestep_loss, laststep_loss, ls_loss, tdf_loss, xdf_loss]
        constraints = []

        loss = PenaltyLoss(objectives, constraints)

        ## Problem
        self.problem = Problem([encoder_x0, encoder_FX, noiseBlock, dynamics_model, decoder_x, decoder_FX], loss)
        self.optimizer = torch.optim.Adam(self.problem.parameters(), lr = self.lr)

        self.problem.show()


    def get_data(self, X_train, U_train, X_dev, U_dev, X_test, U_test):
        nt = X_train.shape[1]
        nx = X_train.shape[-1]
        nu = U_train.shape[-1]

        n_param = X_train.shape[0]
        s = np.random.choice(np.arange(nt, dtype=np.int64), [n_param, self.nBPP], replace=True)

        trainX = np.zeros([n_param, self.nBPP, self.lMB, nx])
        trainU = np.zeros([n_param, self.nBPP, self.lMB, nu])
        
        for i in range(n_param):
            for j in range(self.nBPP):
                if s[i,j]+self.lMB < nt:
                    trainX[i, j] = X_train[i, s[i,j] : s[i,j]+self.lMB]
                    trainU[i, j] = U_train[i, s[i,j] : s[i,j]+self.lMB]
                else:
                    temp1 = nt-s[i,j]
                    temp2 = self.lMB - temp1
                    trainX[i, j, :temp1] = X_train[i, s[i,j]:]
                    trainU[i, j, :temp1] = U_train[i, s[i,j]:]
                    
                    trainX[i, j, temp1:] = X_train[i, :temp2]
                    trainU[i, j, temp1:] = U_train[i, :temp2]


        n_param = X_dev.shape[0]
        s = np.random.choice(np.arange(nt, dtype=np.int64), [n_param, self.nBPP], replace=True)

        devX = np.zeros([n_param, self.nBPP, self.lMB, nx])
        devU = np.zeros([n_param, self.nBPP, self.lMB, nu])
        
        for i in range(n_param):
            for j in range(self.nBPP):
                if s[i,j]+self.lMB < nt:
                    devX[i, j] = X_dev[i, s[i,j] : s[i,j]+self.lMB]
                    devU[i, j] = U_dev[i, s[i,j] : s[i,j]+self.lMB]
                else:
                    temp1 = nt-s[i,j]
                    temp2 = self.lMB - temp1
                    devX[i, j, :temp1] = X_dev[i, s[i,j]:]
                    devU[i, j, :temp1] = U_dev[i, s[i,j]:]
                    
                    devX[i, j, temp1:] = X_dev[i, :temp2]
                    devU[i, j, temp1:] = U_dev[i, :temp2]

        
        trainX = np.concatenate(trainX, axis=0)
        trainX = torch.tensor(trainX, dtype=torch.float32).to(self.device)
        trainU = np.concatenate(trainU, axis=0)
        trainU = torch.tensor(trainU, dtype=torch.float32).to(self.device)
        train_data = DictDataset({'X': trainX, 'x0': trainX[:, 0:1, :],
                                'U': trainU}, name='train')
        train_loader = DataLoader(train_data, batch_size=self.nMB,
                                collate_fn=train_data.collate_fn, shuffle=True)
        
        devX = np.concatenate(devX, axis=0)
        devX = torch.tensor(devX, dtype=torch.float32).to(self.device)
        devU = np.concatenate(devU, axis=0)
        devU = torch.tensor(devU, dtype=torch.float32).to(self.device)
        dev_data = DictDataset({'X': devX, 'x0': devX[:, 0:1, :],
                                'U': devU}, name='val')
        dev_loader = DataLoader(dev_data, batch_size=self.nMB,
                                collate_fn=dev_data.collate_fn, shuffle=True)
        
        testX = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        testU = torch.tensor(U_test, dtype=torch.float32).to(self.device)
        test_data = {'X': testX, 'x0': testX[:, 0:1, :],
                    'U': testU}
        
        return train_loader, dev_loader, test_data


    def train_model(self, output_paths, data_train, ft_train, data_dev, ft_dev, data_test, ft_test):
        train_loader, dev_loader, test_data = \
            self.get_data(data_train, ft_train, data_dev, ft_dev, data_test, ft_test)
        
        callbacker = Callback(self.device)
        logger = BasicLogger(args = None, save_dir = output_paths, verbosity = 1, stdout=['dev_loss', 'train_loss'])

        if self.problem == None:
            raise ValueError("Problem has to be initiated first.")
        
        trainer = Trainer(
            problem = self.problem,
            train_data = train_loader,
            dev_data = dev_loader,
            test_data = test_data,
            optimizer = self.optimizer,
            logger = logger,
            patience = self.patience,
            warmup = self.warmup,
            epochs = self.n_epoch,
            eval_metric = "dev_loss",
            train_metric = "train_loss",
            dev_metric = "dev_loss",
            test_metric = "dev_loss",
            lr_scheduler = self.lr_patience,
            device = self.device,
            callback = callbacker
        )

        best_model = trainer.train()
        self.problem.load_state_dict(best_model)

        return self.problem

