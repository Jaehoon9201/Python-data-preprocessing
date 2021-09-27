#Reference : https://www.youtube.com/watch?v=s2K1JfNR7Sc
# by Steve Brunton

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size':18})

dt = 0.001
t = np.arange(0,1,dt)
f= np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
f_clean = f
f= f+2.5*np.random.randn(len(t))

plt.plot(t,f,color = 'c', LineWidth = 1.5, label =  'Noisy')
plt.plot(t,f_clean,color = 'k', LineWidth = 2, label =  'Clean')
plt.xlim(t[0], t[-1])
plt.savefig('plot.png', dpi = 100)
plt.legend()

n = len(t)
fhat = np.fft.fft(f,n)
PSD = fhat * np.conj(fhat)/n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1,np.floor(n/2), dtype = 'int')

fig, axs = plt.subplots(2,1)

plt.sca(axs[0])
plt.plot(t,f,color = 'c', LineWidth = 1.5, label = 'Noisy')
plt.plot(t,f_clean, color = 'k', Linewidth = 2, label = 'Clean')
plt.xlim(t[0],t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L],PSD[L], color= 'c', LineWidth = 2, label = 'Noisy')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()
plt.show()
