from sklearn import linear_model
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
#from matplotlib.pyplot import plot, show, figure, title
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torchaudio
import time

import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import IPython.display as ipd
import soundfile as sf

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import time
import librosa, librosa.display
from matplotlib.mlab import window_hanning, specgram
from matplotlib.colors import LogNorm
from scipy.io import wavfile

import torchaudio
import torch.nn as nn
import torch
from torchaudio import transforms
import math, random
from scipy.sparse import coo_matrix
import csv
import soundfile as sf
import pandas as pd

import openpyxl
from openpyxl import Workbook
from pathlib import Path

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

df_csv = 'comp_results/test.csv'
names_ = ['0', '1', '2','3','4','5','6','7','8','9','10']
df = pd.read_csv(df_csv, names =names_) #, names =['0', '1', '2','3','4','5','6','7','8','9','10']
df.to_csv('comp_results/test_comp_org.csv')
print(' ============ org ============')
print(df.head())
file_size =Path(r'comp_results/test_comp_org.csv').stat().st_size
print("The org file size is:", file_size,"bytes\n\n\n")


for j in range(len(names_)):
    for i in df.index[df[str(names_[j])] == 0].tolist():
        df.loc[i,str(j)] = ''

print(' ============ remove zeros ============')
print(df.head())
df.to_csv('comp_results/test_comp.csv')
file_size =Path(r'comp_results/test_comp.csv').stat().st_size
print("The compressed file size is:", file_size,"bytes")


