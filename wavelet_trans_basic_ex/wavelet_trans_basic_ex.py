# References
# https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
# https://github.com/taspinar/siml/blob/master/notebooks/WV2%20-%20Visualizing%20the%20Scaleogram%2C%20time-axis%20and%20Fourier%20Transform.ipynb

import os
import pywt
#from wavelets.wave_python.waveletFunctions import *
import itertools
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import torchaudio
import torch.nn as nn
import torch
from torchaudio import transforms

# First lets load the el-Nino dataset, and plot it together with its time-average



class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        # sig = sig[0][0:25000]
        # sig = torch.reshape(sig, (1, 25000))
        # print(sig.shape)
        return (sig, sr)


    def wavelet_transform(signal, waveletname='cmor63.5-2',cmap=plt.cm.viridis):
                                                #waveletname='cmor999-2',

        time = np.arange(0, 100000) * 1 / 50000
        scales = np.arange(2, 130)
        # scales = np.arange(2, 2001)
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        period = 1. / frequencies
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


    def plot_wavelet(signal,
                     waveletname='cmor63.5-2',
                     cmap=plt.cm.viridis,
                     title='Wavelet Transform (Power Spectrum) of signal',
                     ylabel='Period',
                     xlabel='Time'):

        time = np.arange(0, 100000) * 1 / 50000
        # https://buildmedia.readthedocs.org/media/pdf/pywavelets/stable/pywavelets.pdf
        # For the cmor, fbsp and shan wavelets, the user can specify a specific a normalized center frequency. A value of
        # 1.0 corresponds to 1/dt where dt is the sampling period. In other words, when analyzing a signal sampled at 100 Hz,
        # a center frequency of 1.0 corresponds to ~100 Hz at scale = 1. This is above the Nyquist rate of 50 Hz, so for this
        # particular wavelet, one would analyze a signal using scales >= 2.

        scales = np.arange(2, 130)
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        period = 1. / frequencies
        scale0 = 0.000001
        numlevels = 17
        levels = [scale0]

        for i in range(1, numlevels):
            scale0 *= 2
            levels.append(scale0)
        contourlevels = np.log2(levels)

        # FLIPPING
        #period = np.flip(period)
        #power = np.flip(power, axis = 1)


        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)
        print('time length : ' , len(time))
        print('period length : ',len(period))

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)

        yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()

        fig.colorbar(im, orientation="vertical")
        # Low frequency results are shown below on plot
        plt.savefig('wavelet_{}.png'.format(waveletname),
                    dpi=100, bbox_inches='tight')
        plt.close('all')


data_ = AudioUtil.open("sample_aud2.wav")
data, Fsamp = AudioUtil.open("sample_aud2.wav")
data = data.numpy()[0]


AudioUtil.wavelet_transform(data)
AudioUtil.plot_wavelet(data)
