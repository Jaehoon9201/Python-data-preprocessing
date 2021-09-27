import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch import cuda
import numpy as np
import torch
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
import torch.nn.functional as F
import os
import shutil

print(torch.cuda.get_device_name())
print( torch.cuda.is_available())
print(  torch.__version__)

#device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'


names = []
for i in range(0, 125):
    names.append(f'{i}')


data_train = pd.read_csv('Github_set.csv', names = names)

allData_part = []
GenDataset = []

StackedNum = 4
Samp_Num = 12 * StackedNum

TobeStacked_all = []

# 12
TobeStacked = np.array([[0, 0, 3, 0, 0, 3], [0, 0, 3, 0, 0, -3], [0, 0, -3, 0, 0, 3], [0, 0, -3, 0, 0, -3],
                        [0, 3, 0, 0, 3, 0], [0, 3, 0, 0, -3, 0], [0, -3, 0, 0, 3, 0], [0, -3, 0, 0, -3, 0],
                        [3, 0, 0, 3, 0, 0], [3, 0, 0, -3, 0, 0], [-3, 0, 0, 3, 0, 0], [-3, 0, 0, -3, 0, 0]])
TobeStacked = TobeStacked.reshape(-1, 6)
TobeStacked = pd.DataFrame(TobeStacked)
print('TobeStacked : ','\n', TobeStacked, '\n','\n',)

for i in range(0, StackedNum) :
    TobeStacked_all.append(TobeStacked)

TobeStacked_all = pd.concat(TobeStacked_all, axis=0, ignore_index=True)
TobeStacked_all = pd.DataFrame(TobeStacked_all)
print('TobeStacked_all : ','\n', TobeStacked_all)



for i in range(0, 9) :
    dataset = data_train[data_train['124'] == i].sample(n=Samp_Num, replace = True)
    dataset.reset_index(drop=True, inplace=True)
    GenDataset = pd.concat([dataset, TobeStacked_all], axis=1, ignore_index=True)
    print(GenDataset)
    allData_part.append(GenDataset)


output_file = r'Github_gener_ex.csv'
dataCombine = pd.concat(allData_part, axis=0, ignore_index=True)
dataCombine.to_csv(output_file, index=False)
