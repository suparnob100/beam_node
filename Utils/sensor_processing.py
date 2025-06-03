import os
import numpy as np
import pysensors as ps
import numpy.linalg as LA
import matplotlib.pyplot as plt

class sensor_processing:
    def __init__(self, data, config):
        self.data = data
        self.data_reshaped = data.reshape([data.shape[0]*data.shape[1], data.shape[-1]])
        self.seed = config["seed"]
        self.n_SVD_basis = config["sensors"]["n_SVD_basis"]
        self.n_A_basis = config["sensors"]["n_A_basis"]
        self.nx = config["data"]["nx"]
        self.err_cap = 0.00002

    def perform_svd(self):
        self.u, self.s, self.v = LA.svd(self.data_reshaped.T @ self.data_reshaped)

    def plot_singular(self, err_cap = 0.00002):
        self.err_cap = err_cap
        s = np.power(self.s, 0.5)
        s_mass = np.cumsum(s)

        self.s_count = [i for i,ss in enumerate(s_mass) if ss/s_mass[-1]>(1-self.err_cap)][0]+1

        print(f"The preferred number of sensors is: {self.s_count}")

        plt.figure()
        plt.scatter(np.r_[0: len(s): 1], s_mass/np.sum(s))
        plt.show()

        plt.figure()
        plt.semilogy(np.r_[0: len(s): 1], 1-s_mass/np.sum(s), 'o-')
        plt.show()

    def opt_sensor_loc(self, num_sensors = None, fill_gaps = True):
        s = np.power(self.s, 0.5)
        s_mass = np.cumsum(s)
        self.s_count = [i for i,ss in enumerate(s_mass) if ss/s_mass[-1]>(1-self.err_cap)][0]+1

        if num_sensors == None and fill_gaps == False:
            self.ps_model = ps.SSPOR(ps.basis.SVD(self.n_SVD_basis, random_state=self.seed), n_sensors=self.s_count)
            ps_model.fit(self.data_reshaped/np.max(np.abs(self.data_reshaped)), seed=self.seed)
            self.sensor_placement = np.sort(ps_model.get_selected_sensors()[:self.s_count])
        elif num_sensors != None and fill_gaps == False:
            self.ps_model = ps.SSPOR(ps.basis.SVD(self.n_SVD_basis, random_state=self.seed), n_sensors=num_sensors)
            ps_model.fit(self.data_reshaped/np.max(np.abs(self.data_reshaped)), seed=self.seed)
            self.sensor_placement = np.sort(ps_model.get_selected_sensors()[:num_sensors])
        elif num_sensors == None and fill_gaps == True:
            raise Exception("Cannot have gap filling with no preferred sensor count")
        else:
            ps_model = ps.SSPOR(ps.basis.SVD(self.n_SVD_basis, random_state=self.seed), n_sensors=self.s_count)
            ps_model.fit(self.data_reshaped/np.max(np.abs(self.data_reshaped)), seed=self.seed)
            self.sensor_placement = np.sort(ps_model.get_selected_sensors()[:self.s_count])
            extended_subset = np.concatenate(([0], self.sensor_placement, [self.nx]))
            
            while self.s_count < num_sensors:
                gaps = np.diff(extended_subset)
                max_gap_index = np.argmax(gaps)
                point1 = extended_subset[max_gap_index]
                point2 = extended_subset[max_gap_index+1]
                midpoint = (point1 + point2) // 2
                self.sensor_placement = np.sort(np.append(self.sensor_placement, midpoint))
                extended_subset = np.sort(np.append(extended_subset, midpoint))
                self.s_count += 1

        print(f"Sensors are in {self.sensor_placement}")

        # C Matrix
        C_Mat = np.zeros([len(self.sensor_placement), self.nx])
        for i in range(len(self.sensor_placement)):
            C_Mat[i, self.sensor_placement[i]] = 1
        self.C_Mat = C_Mat

        # A Matrix
        A_Mat = ps_model.basis_matrix_[:, :self.n_A_basis]
        self.A_Mat = A_Mat

    def apply_sensors(self, data_train, data_val, data_test):
        RS_train = data_train[:, :, :, self.sensor_placement]
        RS_val = data_val[:, :, :, self.sensor_placement]
        RS_test = data_test[:, :, :, self.sensor_placement]
        
        return RS_train, RS_val, RS_test

    def load(self, path):
        path = os.path.abspath(os.path.join(path, "compression_matrices"))
        self.A_Mat = np.load(os.path.join(path, "A_Mat.npy"))
        self.C_Mat = np.load(os.path.join(path, "C_Mat.npy"))
        self.sensor_placement = np.load(os.path.join(path, "sensor_placement.npy"))

        Theta = self.C_Mat@self.A_Mat
        self.pinv_Theta = np.linalg.pinv(Theta)

        return self.A_Mat, self.C_Mat, self.pinv_Theta, self.sensor_placement

    def save(self, path):
        path = os.path.abspath(os.path.join(path, "compression_matrices"))
        np.save(os.path.join(path, "A_Mat.npy"), self.A_Mat)
        np.save(os.path.join(path, "C_Mat.npy"), self.C_Mat)
        np.save(os.path.join(path, "sensor_placement.npy"), self.sensor_placement)
