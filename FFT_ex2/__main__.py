import matplotlib.pyplot as plt
import numpy as np

from FFT import FFT
from matplotlib.mlab import window_hanning, specgram
from matplotlib.colors import LogNorm


plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size':18})

dt = 0.001
t = np.arange(0,1,dt)
f= np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
f_clean = f
f= f+2.5*np.random.randn(len(t))

'''
def FFT(sample_rate, duration, signal):
    """
    sample_rate (Hz), duration (s), signal: time-domain signal    
    """
    # Number of samples in normalized_tone
    N = int(sample_rate * duration)

    yf = rfft(signal)                    # fft calculates the transform itself. rfft calcaulates only the positive frequency component (real input).
    xf = rfftfreq(N, 1 / sample_rate)    # fftfreq calculates the frequencies in the center of each bin in the output of fft().

    NFFT = N
    Y=np.fft.fft(signal)/NFFT            # fft computing and normaliation
    if len(signal) != 1:
        Y=Y[range(math.trunc(NFFT/2))]   # single sied frequency range
    amplitude_Hz = 2*abs(Y)
    phase_ang = np.angle(Y)*180/np.pi
  
    return xf, yf, phase_ang
'''
m1xf, m1yf, m1angle = FFT(1/dt, 1, f)

plt.figure(figsize = (12,4))
plt.ylabel('yf  ( =rfft(sig) )')
plt.plot(np.abs(m1yf))
plt.grid()
plt.show()
