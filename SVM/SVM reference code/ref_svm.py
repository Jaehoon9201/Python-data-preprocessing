# [Reference] https://gist.github.com/mlaves/c98cd4e6bcb9dbd4d0c03b34bacb0f65
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.datasets import make_moons
sns.set()
np.random.seed(0)

# ========================================
#         Plotting Training Data
# ========================================
x, y = make_moons(20, noise=0.1)
y[np.where(y==0)] = -1

fig, ax = plt.subplots()
ax.scatter(x[np.where(y==-1),0], x[np.where(y==-1),1], label='Class 1')
ax.scatter(x[np.where(y==1),0], x[np.where(y==1),1], label='Class 2')
ax.set_title('Training data')
ax.legend()
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)




# ========================================
#           Defining a Model
# ========================================
#The Hinge Loss
def hinge_loss(x, y):
    return torch.max(torch.zeros_like(y), 1-y*x).mean()

# SVM
class KernelSVM(torch.nn.Module):
    def __init__(self, train_data_x, kernel='rbf',
                 gamma_init=1.0, train_gamma=True):
        super().__init__()
        assert kernel in ['linear', 'rbf']
        self._train_data_x = train_data_x

        if kernel == 'linear':
            self._kernel = self.linear
            self._num_c = 2
        elif kernel == 'rbf':
            self._kernel = self.rbf
            self._num_c = x.size(0)
            self._gamma = torch.nn.Parameter(torch.FloatTensor([gamma_init]),
                                             requires_grad=train_gamma)
        else:
            assert False

        self._w = torch.nn.Linear(in_features=self._num_c, out_features=1)

    def rbf(self, x, gamma=1):
        y = self._train_data_x.repeat(x.size(0), 1, 1)
        return torch.exp(-self._gamma * ((x[:, None] - y) ** 2).sum(dim=2))

    @staticmethod
    def linear(x):
        return x

    def forward(self, x):
        y = self._kernel(x)
        y = self._w(y)
        return y

# define
model_linear = KernelSVM(x, kernel='linear')
model_kernel = KernelSVM(x, kernel='rbf')
opt_linear = torch.optim.SGD(model_linear.parameters(), lr=0.1)
opt_kernel = torch.optim.SGD(model_kernel.parameters(), lr=0.1)

# ========================================
#           Training
# ========================================
for i in range(1000):
    opt_linear.zero_grad()
    opt_kernel.zero_grad()

    pred_linear = model_linear(x)
    loss_linear = hinge_loss(pred_linear, y.unsqueeze(1))
    pred_kernel = model_kernel(x)
    loss_kernel = hinge_loss(pred_kernel, y.unsqueeze(1))

    loss_linear.backward()
    opt_linear.step()
    loss_kernel.backward()
    opt_kernel.step()

# ========================================
#         Training - results
# ========================================
print("loss linear model", loss_linear.item())
print("loss kernel model", loss_kernel.item())

grid_x, grid_y = torch.meshgrid(torch.arange(x.min()*1.1, x.max()*1.1, step=0.1),
                                torch.arange(x.min()*1.1, x.max()*1.1, step=0.1))
x_test = torch.stack((grid_x, grid_y)).reshape(2, -1).transpose(1,0)

y_test_linear = model_linear(x_test).detach()
y_test_kernel = model_kernel(x_test).detach()

y_test_linear = y_test_linear.transpose(1,0).reshape(grid_x.shape).numpy()
y_test_kernel = y_test_kernel.transpose(1,0).reshape(grid_x.shape).numpy()



fig, ax = plt.subplots(1,2, figsize=(8,3))
plt.rcParams['axes.grid'] = False

cs0 = ax[0].contourf(grid_x.numpy(), grid_y.numpy(), y_test_linear)
ax[0].contour(cs0, '--', levels=[0], colors='tab:green', linewidths=2)
ax[0].plot(np.nan, label='decision boundary', color='tab:green')
ax[0].scatter(x[np.where(y==-1),0], x[np.where(y==-1),1])
ax[0].scatter(x[np.where(y==1),0], x[np.where(y==1),1])
ax[0].legend()
ax[0].set_title('Linear Kernel')

cs1 = ax[1].contourf(grid_x.numpy(), grid_y.numpy(), y_test_kernel)
cs11 = ax[1].contour(cs1, '--', levels=[0], colors='tab:green', linewidths=2)
ax[1].plot(np.nan, label='decision boundary', color='tab:green')
ax[1].scatter(x[np.where(y==-1),0], x[np.where(y==-1),1])
ax[1].scatter(x[np.where(y==1),0], x[np.where(y==1),1])
ax[1].set_title('RBF Kernel')

fig.subplots_adjust(wspace=0.2, hspace=0.1,right=0.8)
cbar_ax = fig.add_axes([0.82, 0.13, 0.02, 0.67])
cbar = fig.colorbar(cs1, cax=cbar_ax, )
cbar.add_lines(cs11)

plt.show()