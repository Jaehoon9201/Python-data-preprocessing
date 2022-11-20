# Ref : https://www.kaggle.com/code/anjum48/continuous-wavelet-transform-cwt-in-pytorch
import pywt
import numpy as np
import matplotlib.pyplot as plt
import torchaudio

class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)


# ==============================================
#                     Para set
# ==============================================

data, Fsamp = AudioUtil.open("TRAIN M1 D40 L0 LABEL0 Num170.wav")
MaxIdx = 100
dataprocess = 'DataCut'  # ReSampl / DataCut
waveletname = 'cmor63.5-2'
# waveletname = 'mexh'


# ==============================================
#              data process method
# ==============================================
if dataprocess == 'ReSampl' :
    # ver 1. resampling with period setting
    scales = np.arange(2, 130)
    time = np.arange(0, 100000) * 1 / 50000
    ResmapleIdx = []
    for i in range(0, MaxIdx):
        ResmapleIdx.append(int(i * 100000/MaxIdx))
    time = time[ResmapleIdx]
    data = data.numpy()[0][ResmapleIdx]
    dt = time[1] - time[0]

elif dataprocess == 'DataCut' :
    # ver 2. data cut without adjusting resampling
    scales = np.arange(2, 130)
    time = np.arange(0, 100000) * 1 / 50000
    DataCutIdx = MaxIdx + 1
    time = time[0:DataCutIdx]
    data = data.numpy()[0][0:DataCutIdx]
    dt = time[1] - time[0]




# ==============================================
#                 wavelet method
# ==============================================
if waveletname == 'mexh':
    '''
    Not Complex cwtmatr. --> direct plotting is possible !
    '''
    cwtmatr, freqs = pywt.cwt(data, scales,waveletname, dt)
    plt.imshow(cwtmatr, cmap='viridis', aspect='auto',
                 vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.savefig('wavelet_{}.png'.format(waveletname),
                dpi=100, bbox_inches='tight')
    plt.close('all')

elif waveletname == 'cmor63.5-2':
    '''
    Complex transforms
    Ideally we want to create an image that relates frequency and time. 
    One way of doing this is to use a complex transform, and take the magnitude of the real and imaginary parts. 
    Below we'll use the complex Morlet wavelet in SciPy which looks like this:
    '''

    cwtmatr, freqs = pywt.cwt(data, scales,waveletname, dt)  # cmor63.5-2    mexh
    power = (abs(cwtmatr)) ** 2
    period = 1. / freqs
    scale0 = 0.000001
    numlevels = 17
    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
    contourlevels = np.log2(levels)

    # Data
    wt_freq_results = np.log2(period)
    wt_power_results = np.log2(power)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(time, wt_freq_results, wt_power_results, contourlevels, extend='both', cmap='viridis')

    yticks = np.arange(period.min(), period.max())
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    # ax.invert_yaxis()

    fig.colorbar(im, orientation="vertical")
    # Low frequency results are shown below on plot
    plt.savefig('wavelet_{}.png'.format(waveletname),
                dpi=100, bbox_inches='tight')
    plt.close('all')