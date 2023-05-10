# ---------------------------------
#           Reference
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.02-Discrete-Fourier-Transform.html
# ---------------------------------

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')

# sampling rate
sr = 60
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 0.
x = 1
freq = 1.
x += 2*np.sin(2*np.pi*freq*t)
freq = 2
x += 3*np.sin(2*np.pi*freq*t)
freq = 3
x += 4*np.sin(2*np.pi*freq*t)
freq = 29
x += 5*np.sin(2*np.pi*freq*t)


plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')

plt.show()

# -------------------
def DFT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)

    return X

X = DFT(x)

# calculate the frequency
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T
print(freq)
plt.figure(figsize = (8, 6))
plt.stem(freq, abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.show()



# -------------------
n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]

# normalize the amplitude
X_mag =abs(X[:n_oneside]) * 2/N
X_mag[0] =X_mag[0]/2
# X_mag[n_oneside-1] =X_mag[n_oneside-1]/2

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(f_oneside, (X_mag), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')

plt.subplot(122)
plt.stem(f_oneside, (X_mag), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.xlim(-1, 10)
plt.tight_layout()
plt.show()

print(X_mag)
