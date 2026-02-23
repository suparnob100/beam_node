import numpy as np
from scipy.signal import welch
from scipy.signal.windows import hann
import matplotlib.pyplot as plt

def psd_custom(t, ts):    
    
    dt = t[1]-t[0]
    fs = 1/dt
    
    nblock = len(ts)
    overlap = 1024
    win = hann(nblock)

    F, Pxx = welch(x=ts, fs=fs, window=win,noverlap=overlap,nfft=nblock,detrend=False,return_onesided=True)    
    Pxx = np.multiply(10, np.log10(Pxx))
    
    return F, Pxx


def psd_cutoff(t,ts,f_cut):
    
    dt = t[1]-t[0]
    fs = 1/dt   
    
    m=int(np.floor(0.5*fs/f_cut))
    lenC=int(np.floor(len(ts)/m))

    tn=np.linspace(0, t_end, num=lenC, endpoint=False)
    # tn=tn[np.r_[0:lenC].astype(int)]
    
    dtn=tn[-1]-tn[-2]
    fsn=1/dtn

    for loop in range(m):

        tmpvar=ts[loop::m]

        if loop == 0:
            nblock = lenC #len(tmpvar[0:lenC])
            overlap = 0
            win = hann(nblock)
            F,Px = welch(x=tmpvar[0:lenC],fs=fsn, window=win,noverlap=overlap,nfft=nblock,detrend=False,return_onesided=True)
            Pxx = Px
            # Px=np.zeros((m,int(np.ceil(len(tmpvar[0:lenC]))/2)))
           
        else:
            nblock = len(tmpvar[0:lenC])
            win = hann(nblock)
            F,Px = welch(x=tmpvar[0:lenC],fs=fsn, window=win,noverlap=overlap,nfft=nblock,detrend=False,return_onesided=True)
            Pxx = np.vstack([Pxx, Px])
                              
    
    if m > 1:
        Pxx=np.mean(Pxx,axis=0)
    else:
        Pxx=Px
           
    
    Pxx = np.multiply(10, np.log10(Pxx))
    
    return F, Pxx

dt = 0.01
fs = 1/dt
t_end = 10000

t=np.linspace(0, t_end, num=t_end*int(fs), endpoint=False)
y=np.sin(2*np.pi*5*t)

F,Pxx = psd_cutoff(t, y,10)
plt.plot(F,Pxx)
