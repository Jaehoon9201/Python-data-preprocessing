import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import sys
from matplotlib import animation
import os

#make_dir(path="model")
def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


cond = 'generated_file'
data_num = 100

make_dir(path="%s" %(cond))
file_1 = open("%s/test1.txt" %(cond), "w");
file_2 = open("%s/test2.txt" %(cond), "w");

def random_data_generator(data_num):
    return np.random.rand(data_num)

data1 = random_data_generator(100)
data2 = random_data_generator(100)


data1 = list((np.round(data1, 4)))
data2 = list((np.round(data2, 4)))

print('data1 : \n', data1)
print('\n\ndata2 : \n', data2)


for i in range(len(data1)):
    if i % 10 == 0 and i > 0:
        file_1.write('\n')
        file_1.write(',')
        file_2.write('\n')
        file_2.write(',')
    elif i % 10 != 0 and i > 0:
        file_1.write(',')
        file_2.write(',')

    file_1.write(str(data1[i]))
    file_2.write(str(data2[i]))

file_1.close()

file_2.close()
