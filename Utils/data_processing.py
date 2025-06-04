import os
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(path, normalize = False, disp_norm = None, vel_norm = None, ft_norm = None):
    snapshot_data = np.load(path + "/snapshot_data.npy")
    parameters = np.load(path + "/params.npy")
    forcing = np.load(path + "/forcing.npy")

    if disp_norm is not None and normalize == True:
        snapshot_data[:, 0] /= disp_norm
        snapshot_data[:, 1] /= vel_norm
    else:
        disp_norm = np.max(np.abs(snapshot_data[:, 0]))
        vel_norm = np.max(np.abs(snapshot_data[:, 1]))
        if normalize == True:
            snapshot_data[:, 0] /= disp_norm
            snapshot_data[:, 1] /= vel_norm

    if ft_norm is not None and normalize == True:
        for col in range(forcing.shape[-1]):
            forcing[:, :, col] = (forcing[:, :, col] - ft_norm[col, 1])/(ft_norm[col, 0] - ft_norm[col, 1])
    else:
        ft_norm = np.zeros([forcing.shape[-1], 2])
        for col in range(forcing.shape[-1]):
            ft_norm[col, 0] = np.max(forcing[:, :, col])
            ft_norm[col, 1] = np.min(forcing[:, :, col])
            if normalize == True:
                forcing[:, :, col] = (forcing[:, :, col] - ft_norm[col, 1])/(ft_norm[col, 0] - ft_norm[col, 1])
        
    return snapshot_data, parameters, forcing, disp_norm, vel_norm, ft_norm

def save_dataset(path, data, params, forcing, cluster = None, disp_norm = None, vel_norm = None, ft_norm = None):
    if disp_norm is not None:
        data[:, 0] *= disp_norm
        data[:, 1] *= vel_norm
        for col in range(forcing.shape[-1]):
            forcing[:, col] *= (ft_norm[col, 0] - ft_norm[col, 1])
            forcing[:, col] += ft_norm[col, 1]

    if cluster is not None:
        if not os.path.exists(os.path.join(path, f"{cluster}")):
            os.makedirs(os.path.join(path, f"{cluster}"))         
        np.save(os.path.join(path, f"{cluster}/snapshot_data.npy"), data)
        np.save(os.path.join(path, f"{cluster}/params.npy"), params)
        np.save(os.path.join(path, f"{cluster}/forcing.npy"), forcing)
    else:
        np.save(os.path.join(path, f"snapshot_data.npy"), data)
        np.save(os.path.join(path, f"params.npy"), params)
        np.save(os.path.join(path, f"forcing.npy"), forcing)

def load_cluster(path, cluster, normalize = False, disp_norm = None, vel_norm = None, ft_norm = None):
    path = os.path.join(path, str(cluster))
    snapshot_data, parameters, forcing, disp_norm, vel_norm, ft_norm = load_dataset(path, normalize, disp_norm, vel_norm, ft_norm)
    return snapshot_data, parameters, forcing, disp_norm, vel_norm, ft_norm

def save_cluster(path, cluster, data, params, forcing, disp_norm = None, vel_norm = None, ft_norm = None):
    if not os.path.exists(os.path.join(path, str(cluster))):
        os.makedirs(os.path.join(path, str(cluster)))
    save_dataset(path, data, params, forcing, cluster, disp_norm, vel_norm, ft_norm)

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