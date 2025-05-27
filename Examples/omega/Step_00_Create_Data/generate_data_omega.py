#!/home/halsali/.conda/envs/neuromancer-gpu/bin/python
# coding: utf-8

#%% Initialize
import numpy as np
import argparse
import os 
import sys

import matplotlib.pyplot as plt

if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

utils_dir = os.path.abspath(os.path.join(script_dir, "..", "Utils"))
output_dir = os.path.abspath(os.path.join(script_dir, "..", "Outputs", "Step00", "datasets"))

sys.path.append(utils_dir)


T = np.array([(2/((4**2)*np.pi), 2/np.pi)])  # Period of the vibration range
tau_param = np.min(T)/5 # Impulse width
delta_s = 1/(np.sqrt(2)) # Impulse location
omega = 2*np.pi/(T) # Omega range
max_dt = tau_param/5 # Maximum time step

param_ranges = [omega]

nt = int(np.max(T) / max_dt) + 1 # number of time steps
nx = 101 # number of spatial points
t = np.linspace(0, max_dt * (nt - 1), nt) # Time
x = np.linspace(0, 1, nx) # Space


#%% Define Functions
# Spatial basis function
def phi_k(x, k):
    return np.sqrt(2) * np.sin(k * np.pi * x)


# Modal coefficients (ai) based on the given formula
def a_i(i, s, ep, tau, T, c_v, c_m, t, omega, k_range):

    omega_i = (np.pi*i)**2
    
    # Cik and Dik constants based on dk, ak, and ci
    def CDi_k(dk, ak, fi, omega, ci, k):

        Ci_k = (dk * ak * fi) / ((k * ci * omega)**2 + ak**2)
        Di_k = (k * dk * omega * ci * fi) / ((k * ci * omega)**2 + ak**2)

        return Ci_k, Di_k

    # Define dk based on the given cases
    def d_k(k, T, tau):
        if T == k * tau:
            return (-1)**k / T
        else:
            return (2 * (T**3 * np.cos(np.pi * k) * np.sin(np.pi * k * tau / T))) / \
                (T * (np.pi * k * tau * T**2 - np.pi * k**3 * tau**3))

    # Define fi integral
    def f_i(i, ep, s):
        if i*ep == 2:
            return (1 / np.sqrt(2)) * np.sin(i *s* np.pi)
        else:
            return (8 * np.sqrt(2) * np.sin(i * s * np.pi) * np.sin(i * np.pi * ep / 2)) / (4 * np.pi * i * ep - i**3 * ep**3 * np.pi)


    # Define ci based on the formula
    def c_i(omega_i, c_v, c_m):
        return c_v + c_m * omega_i**2


    # Update alpha_k
    def alpha_k(omega_i, k, omega):
        return omega_i**2 - (k * omega)**2
    
    
    dk = [d_k(k+1, T, tau) for k in k_range]
    alphak =  [alpha_k(omega_i, k+1, omega) for k in k_range]


    ci = c_i(omega_i, c_v, c_m)
    fi = f_i(i, ep, s)

    sum_term_d = 0
    sum_term_v = 0

    for k in k_range:
        
        Ci_k, Di_k = CDi_k(dk[k], alphak[k], fi, omega, ci, k)

        sum_term_d += Ci_k * np.cos((k+1) * omega * t)     +     Di_k * np.sin((k+1) * omega * t)

        sum_term_v += (-(k+1) * omega) * Ci_k * np.sin((k+1) * omega * t)    +   ((k+1) * omega) * Di_k * np.cos((k+1) * omega * t)


    return (fi / (omega_i**2 * T)) + sum_term_d, sum_term_v


# Steady State
def get_steady_state(tau, s=1/2**0.5, ep=0.02, omega=0, t=None):
    i_range = range(1, 250)  # Mode numbers
    k_range = range(0, 90)  # Fourier numbers

    # Define constants c_v and c_m
    c_v = 1.0
    c_m = 0.001

    T = 2*np.pi/omega

    W = 0
    W_dot =  0

    for i in i_range:

        ai,ai_dot = a_i(i, s, ep, tau, T, c_v, c_m, t, omega, k_range)
        phi_i= phi_k(x, i)

        W += phi_i.reshape(-1,1) @ ai.reshape(1,-1)
        W_dot += phi_i.reshape(-1,1) @ ai_dot.reshape(1,-1)

    return W[:, :-1], W_dot[:, :-1]


# Sobol Distribution
def generate_sobol(dimensions, num_points, bounds):
    """
    Generates a Sobol sequence.
    """
    from scipy.stats.qmc import Sobol

    sobol = Sobol(d=dimensions)
    samples = sobol.random_base2(m=int(np.log2(num_points)))
    scaled_samples = np.empty_like(samples)
    
    for i in range(dimensions):
        lower, upper = bounds[i]
        scaled_samples[:, i] = samples[:, i] * (upper - lower) + lower
        
    return scaled_samples


def save_data(SS_init, SS_train, SS_val, SS_test, param_init, param_train, param_val, param_test):
    """
    Save Data Into Outputs/Step00/datasets
    """
    np.save(os.path.join(output_dir, f"Init/params.npy"), param_init)
    np.save(os.path.join(output_dir, f"Init/snapshot_data"), SS_init.transpose(0,1,3,2))
    print("Saved Init into " + os.path.join(output_dir, f"Init"))
    np.save(os.path.join(output_dir, f"Train/params.npy"), param_train)
    np.save(os.path.join(output_dir, f"Train/snapshot_data"), SS_train.transpose(0,1,3,2))
    print("Saved Training into " + os.path.join(output_dir, f"Train"))
    np.save(os.path.join(output_dir, f"Val/params.npy"), param_val)
    np.save(os.path.join(output_dir, f"Val/snapshot_data"), SS_val.transpose(0,1,3,2))
    print("Saved Val into " + os.path.join(output_dir, f"Val"))
    np.save(os.path.join(output_dir, f"Test/params.npy"), param_test)
    np.save(os.path.join(output_dir, f"Test/snapshot_data"), SS_test.transpose(0,1,3,2))
    print("Saved Test into " + os.path.join(output_dir, f"Test"))


def create_data(train_batch):
    """
    Create Datasets for Init, Train, Val, Test datasets
    """
    num_init = train_batch//4
    num_train = train_batch
    num_val = train_batch//4
    num_test = train_batch*4

    param_init = generate_sobol(2, num_init, param_ranges)
    param_train = generate_sobol(2, num_train, param_ranges)
    param_val = generate_sobol(2, num_val, param_ranges)
    param_test = generate_sobol(2, num_test, param_ranges)

    SS_init = np.zeros([num_init, 2, nx, nt])
    SS_train = np.zeros([num_train, 2, nx, nt])
    SS_val = np.zeros([num_val, 2, nx, nt])
    SS_test = np.zeros([num_test, 2, nx, nt])

    for i in range(num_init):
        print(f"Init - {i+1}/{num_init}")
        SS_init[i, 0], SS_init[i, 1] = get_steady_state(tau = param_init[i, 0], s = param_init[i, 1], omega = omega, t = t)
    for i in range(num_train):
        print(f"Train - {i+1}/{num_train}")
        SS_train[i, 0], SS_train[i, 1] = get_steady_state(tau = param_train[i, 0], s = param_train[i, 1], omega = omega, t = t)
    for i in range(num_val):
        print(f"Val - {i+1}/{num_val}")
        SS_val[i, 0], SS_val[i, 1] = get_steady_state(tau = param_val[i, 0], s = param_val[i, 1], omega = omega, t = t)
    for i in range(num_test):
        print(f"Test - {i+1}/{num_test}")
        SS_test[i, 0], SS_test[i, 1] = get_steady_state(tau = param_test[i, 0], s = param_test[i, 1], omega = omega, t = t)

    return SS_init, SS_train, SS_val, SS_test, param_init, param_train, param_val, param_test


def main(train_batch):
    print(f"Creating Data for Batch Number - {train_batch}")
    SS_init, SS_train, SS_val, SS_test, param_init, param_train, param_val, param_test = create_data(train_batch)
    print("Saving Data")
    save_data(SS_init, SS_train, SS_val, SS_test, param_init, param_train, param_val, param_test)
    print("Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=16,
                        help='Number of Training Parameters')

    args = parser.parse_args()
    
    main(train_batch = args.batch)