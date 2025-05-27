#%% Initialize
import numpy as np
from upsampler import fourier_upsample_add

class beam_problem:
    def __init__(self, nx = 101, nt = 200, c_v = 1.0, c_m = 0.001, ep = 0.02, i_range = range(1, 250), k_range = range(0, 90), upsample = None, max_dt = None):
        self.nx = nx
        self.nt = nt
        self.c_v = c_v
        self.c_m = c_m
        self.ep = ep
        self.i_range = i_range
        self.k_range = k_range
        self.upsample = upsample
        self.max_dt = max_dt


    def phi_k(self, x, k):
        return np.sqrt(2) * np.sin(k * np.pi * x)
    

    def a_i(self, i, s, ep, tau, T, c_v, c_m, t, omega):

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
        
        
        dk = [d_k(k+1, T, tau) for k in self.k_range]
        alphak =  [alpha_k(omega_i, k+1, omega) for k in self.k_range]


        ci = c_i(omega_i, c_v, c_m)
        fi = f_i(i, ep, s)

        sum_term_d = 0
        sum_term_v = 0

        for k in self.k_range:
            
            Ci_k, Di_k = CDi_k(dk[k], alphak[k], fi, omega, ci, k)

            sum_term_d += Ci_k * np.cos((k+1) * omega * t)     +     Di_k * np.sin((k+1) * omega * t)

            sum_term_v += (-(k+1) * omega) * Ci_k * np.sin((k+1) * omega * t)    +   ((k+1) * omega) * Di_k * np.cos((k+1) * omega * t)

        return (fi / (omega_i**2 * T)) + sum_term_d, sum_term_v
    

    def solve(self, tau, s, omega):
        T = 2*np.pi/omega
        t = np.linspace(0, T, self.nt+1) # Time
        x = np.linspace(0, 1, self.nx) # Space
        
        W = 0
        W_dot = 0

        for i in self.i_range:
            ai, ai_dot = self.a_i(i, s, self.ep, tau, T, self.c_v, self.c_m, t, omega)
            phi_i= self.phi_k(x, i)

            W += phi_i.reshape(-1,1) @ ai.reshape(1,-1)
            W_dot += phi_i.reshape(-1,1) @ ai_dot.reshape(1,-1)

        if self.upsample != None:
            W = W[:, :int(T/self.max_dt)]
            W_dot = W_dot[:, :int(T/self.max_dt)]
            W_ret = np.zeros([self.nx, t.shape[0]+1])
            W_dot_ret = np.zeros([self.nx, t.shape[0]+1])

            for i in range(self.nx):
                W_ret[i] = fourier_upsample_add(W[i], t.shape[0]+1 - W.shape[1])
                W_dot_ret[i] = fourier_upsample_add(W_dot[i], t.shape[0]+1 - W.shape[1])
            
            return W_ret[:, :-1].T, W_dot_ret[:, :-1].T
    
        else:
            return W[:, :-1].T, W_dot[:, :-1].T