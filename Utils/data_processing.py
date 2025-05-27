import numpy as np
import os

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