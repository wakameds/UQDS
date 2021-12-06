import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

#make 40 separated points
X, y = make_blobs(n_samples=40, centers=2, cluster_std=1)

#fit tje model, don't regularize for illustration purposes
my_svm = svm.SVC(kernel='rbf', C=1)
my_svm.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

#plot decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX  = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = my_svm.decision_function(xy).reshape(XX.shape)

#plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles = ['--','-', '--'])
plt.show()