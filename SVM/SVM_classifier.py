
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import mglearn

X, y = make_blobs(n_samples=50, centers=2, random_state=604)

clf = svm.SVC(kernel='linear')
# (kernel='rbf', C=1,gamma=0)
clf.fit(X, y)

# Plot sample data
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)
# Draw Hyper-Planes
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


xx_ = np.linspace(xlim[0], xlim[1], 100)
yy_ = np.linspace(ylim[0], ylim[1], 100)

yy, xx = np.meshgrid(yy_, xx_)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
z = clf.decision_function(xy)
#z = clf.predict(xy)
z = z.reshape(xx.shape)

ax.contour(xx, yy, z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
# Draw Support Vectors
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=60, facecolors='r')

plt.title('SVC with ---- kernel')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()



# plt.figure(figsize=[8,6])
# mglearn.plots.plot_2d_classification(clf, X, eps=0.5, cm='ocean')
# mglearn.discrete_scatter(X[:,0], X[:,1], y)
# plt.show()
