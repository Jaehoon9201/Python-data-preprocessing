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
plt.show()