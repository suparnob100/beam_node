import os
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(path, disp_norm = None, vel_norm = None):
    snapshot_data = np.load(path + "/snapshot_data.npy")
    parameters = np.load(path + "/params.npy")

    if disp_norm:
        snapshot_data[:, 0] /= disp_norm
        snapshot_data[:, 1] /= vel_norm
        return snapshot_data, parameters
    else:
        return snapshot_data, parameters


def save_dataset(path, data, params, cluster=None):
    if cluster:
        if not os.path.exists(os.path.join(path, f"{cluster}")):
            os.makedirs(os.path.join(path, f"{cluster}"))         
        np.save(os.path.join(path, f"{cluster}/snapshot_data.npy"), data)
        np.save(os.path.join(path, f"{cluster}/params.npy"), params)
    else:
        np.save(os.path.join(path, f"snapshot_data.npy"), data)
        np.save(os.path.join(path, f"params.npy"), params)

def load_clusters():
    pass

def save_clusters():
    pass

def parameter_plot(param_train, param_val, param_test):
    plt.figure(figsize=(6, 4))

    if param_train.shape[1] == 1:
        x_train = np.zeros(param_train.shape[0])
        x_val = np.ones(param_val.shape[0])
        x_test = np.full(param_test.shape[0], 2)
        
        plt.scatter(x_train, param_train, label="Train", alpha=0.6)
        plt.scatter(x_val, param_val, label="Val", alpha=0.6)
        plt.scatter(x_test, param_test, label="Test", alpha=0.6)
        
        plt.xticks([0, 1, 2], ['Train', 'Val', 'Test'])
        plt.ylabel("Parameter Value")
        plt.title("Parameter Distribution")
        plt.legend()
        plt.grid(True)

    elif param_train.shape[1] == 2:
        plt.scatter(param_train[:, 0], param_train[:, 1], label="Train", alpha=0.6)
        plt.scatter(param_val[:, 0], param_val[:, 1], label="Val", alpha=0.6)
        plt.scatter(param_test[:, 0], param_test[:, 1], label="Test", alpha=0.6)
        
        plt.ylabel("Parameter 1")
        plt.title("Parameter 2")
        plt.legend()
        plt.grid(True)
    else:
        print("Only supports 1D/2D parameters for now.")
    plt.show()