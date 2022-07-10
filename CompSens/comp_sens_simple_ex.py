

from sklearn import linear_model
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
#from matplotlib.pyplot import plot, show, figure, title
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torchaudio
import time
class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

CompSensNum  = 5000
data, Fsamp = AudioUtil.open("sample_aud.wav")
TimeEnd = 0.5
SampleNum = int(TimeEnd*Fsamp)
t = np.linspace(0, TimeEnd, SampleNum)
data = data.numpy()[0]
data = data[0:SampleNum]


plt.figure(figsize=(12, 3))
plt.plot(t, data, linewidth=0.2, markersize=1)
plt.title('original audio')
plt.grid()
plt.tight_layout()


data = np.reshape(data, (len(data), 1))
CompSensIdx = np.random.randint(0, SampleNum, (CompSensNum,))
CompSensIdx = np.sort(CompSensIdx)
CompSensData = data[CompSensIdx]


plt.figure(figsize=(12, 3))
plt.plot(t, data, 'b', t[CompSensIdx], CompSensData, 'r.', linewidth=0.2, markersize=1)
plt.title('original audio and compressed sensing')
plt.grid()
plt.tight_layout()


Basis = dct(np.eye(SampleNum))
SampBasis = Basis[CompSensIdx, :]
lasso = linear_model.Lasso(alpha=0.001)
lasso.fit(SampBasis, CompSensData.reshape((CompSensNum,)))
ReconsData = idct(lasso.coef_.reshape((SampleNum, 1)), axis=0)
# plt.figure(figsize=(12, 3))
# plt.grid()
# plt.plot(lasso.coef_.reshape((SampleNum, 1)),'b*', markersize=1)

plt.figure(figsize=(12, 3))
plt.plot(t, data, 'b', linewidth=0.2, label = 'orgin')
plt.plot(t, ReconsData, 'k', linewidth=0.2, label = 'recon')
plt.title('original audio and reconstucted audio')
plt.legend()
plt.grid()
plt.tight_layout()


plt.show()
