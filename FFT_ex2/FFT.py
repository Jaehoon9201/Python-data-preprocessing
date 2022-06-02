
from scipy.fft import rfft, rfftfreq
import math
import numpy as np

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
        Y=Y[range(math.trunc(NFFT/2))]       # single sied frequency range
    amplitude_Hz = 2*abs(Y)
    phase_ang = np.angle(Y)*180/np.pi

    return xf, yf, phase_ang