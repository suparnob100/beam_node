import numpy as np
import scipy.signal as signal

#%% Low Pass Filter
def LPF(time, data):
    # Filter specifications
    order = 2000  # Filter order
    Fpass = 60  # Cutoff frequency (Hz) for the low-pass filter
    Fs = 1/(time[1] - time[0])  # Sampling frequency (Hz)
     
    # Frequencies for the filter design
    # Wp is the passband edge frequency and Wst is the stopband edge frequency
    frequencies = [0, Fpass, Fpass + 5, Fs / 2]
    gains = [1, 1, 0, 0]  # Desired gain at the specified frequencies
     
    # Design the FIR low-pass filter using firwin2
    b = signal.firwin2(order + 1, frequencies, gains, fs=Fs)
    
    datafiltered = signal.filtfilt(b, 1, data, axis=0)  # Apply the filter
    
    return datafiltered


class noise_reducer:
    def __init__(self):
        pass

    def apply_LPF(self, signal, desired_cycles):
        for i in range(signal.shape[0]):
            data = data_avg = np.tile(data_avg, desired_cycles, 1)
        return signal



    def signal_averaging(self, signal, nt, num_cycles):
        for i in range(signal.shape[0]):
            data = signal[i]
            data_avg = []
            for cycle in range(num_cycles):
                data_avg.append(data[nt*cycle: nt*(cycle+1)])

            data_avg = np.array(data_avg)
            data_avg = np.mean(data_avg, axis=0)

            signal[i] = data_avg
        return signal
