
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch import cuda
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

device = 'cuda' if cuda.is_available() else 'cpu'


Epoch_num = 20
batch_size = 100
train_loss_values = np.zeros([Epoch_num - 1])
test_loss_values = np.zeros([Epoch_num - 1])

# ========================================
#           Preparing Dataset
# ========================================
class Train_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/train_freezed.csv',delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]

        self.x_data = from_numpy(xy[:, :-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Test_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/test_freezed.csv',delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, :-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_dataset = Train_Dataset()
test_dataset = Test_Dataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=len(Train_Dataset()),
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=len(Train_Dataset()),
                         shuffle=False)

for batch_idx, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device=device).float(), labels.to(device=device).float()
x_tr = inputs
y_tr = labels

for batch_idx, (inputs, labels) in enumerate(test_loader):
    inputs, labels = inputs.to(device=device).float(), labels.to(device=device).float()
x_te = inputs
y_te = labels


# ========================================
#               Training
# ========================================
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(x_tr, y_tr)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_tr, y_tr)

# ========================================
#               Testing
# ========================================
poly_pred = poly.predict(x_te)
rbf_pred = rbf.predict(x_te)

# ========================================
#               Eval
# ========================================
poly_accuracy = accuracy_score(y_te, poly_pred)
poly_f1 = f1_score(y_te, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

rbf_accuracy = accuracy_score(y_te, rbf_pred)
rbf_f1 = f1_score(y_te, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))