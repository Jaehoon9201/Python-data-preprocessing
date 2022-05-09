import numpy as np
import sklearn.datasets, sklearn.decomposition
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from skimage import color


X = cv2.imread("Lenna.png",0)

mu = np.mean(X, axis=0)
pca = sklearn.decomposition.PCA()
pca.fit(X)


width=20
height=5
rows = 1
cols = 5
axes=[]
fig=plt.figure()

for a in range(rows*cols):

    nComp = a*4
    if nComp == 0 : nComp = 1

    Xhat = np.dot(pca.transform(X)[:, :nComp], pca.components_[:nComp, :])
    Xhat += mu

    axes.append( fig.add_subplot(rows, cols, a+1) )
    subplot_title=("# of principal components : %d") %(nComp)
    axes[-1].set_title(subplot_title)

    plt.imshow(Xhat)

fig.tight_layout()
plt.show()
