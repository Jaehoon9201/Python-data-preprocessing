import numpy as np
import pywt
dt = 0.01 # 100 Hz sampling
fs = 1 / dt
print('\nSampling Frequency : ', 1/dt)

#==============================
#     Set Target Frequency
#==============================
TargetFreq = [100, 50, 33.33333333, 25]
print('\nSet Target Frequency : ', TargetFreq)

frequencies = np.array(TargetFreq) / fs # normalize
scale = pywt.frequency2scale('cmor1.5-1.0', frequencies)
print('SCALE of cmor1.5-1.0 :', scale)

scale = pywt.frequency2scale('cmor1.5-2.0', frequencies)
print('SCALE of cmor1.5-2.0 :', scale)

#==============================
#     Set Target Scale
#==============================
TargetScale = [1, 2, 3, 4]
print('\nSet Target Frequency : ', TargetScale)

frequencies = pywt.scale2frequency('cmor1.5-1.0', TargetScale) / dt
print('FREQUENCY of cmor1.5-1.0 :', frequencies)

frequencies = pywt.scale2frequency('cmor1.5-2.0', TargetScale) / dt
print('FREQUENCY of cmor1.5-2.0 :', frequencies)

frequencies = pywt.scale2frequency('cmor1.5-0.5', TargetScale) / dt
print('FREQUENCY of cmor1.5-0.5 :', frequencies)