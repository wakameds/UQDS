
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset and subtract the column means from each row value.
data = pd.read_csv("pca_toy.txt", sep="\s+", names=["x","y"])


#(a) Mean center the data, i.e. subtract the means from the two columns.
X = data - data.mean() 
plt.scatter(X["x"], X["y"], marker="o")
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.show()


#(b) Next calculate the covariance matrix of the mean centered data.
X = X.to_numpy(dtype = "float")
cov = np.matmul(X.T , X)


#(c) Find the eigenvalues and corresponding eigenvectors of the covariance matrix. Plot the eigen
# vectors along with: the original data and then with the mean centered data.
e_values, e_vectors = np.linalg.eig(cov)

#Set origin as the center of the data
ox = data["x"].mean()
oy = data["y"].mean()

#Plot eigenvectors along the original data
fig, ax = plt.subplots()
plt.scatter(data["x"], data["y"])
plt.quiver(ox, oy, e_vectors[:,0][0] , e_vectors[:,0][1], color=['r'], scale=5)
plt.quiver(ox, oy, e_vectors[:,1][0] , e_vectors[:,1][1], color=['g'], scale=8)
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()


#Plot eigenvectors along the mean centered data
fig, ax = plt.subplots()
plt.scatter(X[:,0], X[:,1])
plt.quiver(0,0, e_vectors[:,0][0] , e_vectors[:,0][1], color=['r'], scale=5)
plt.quiver(0,0, e_vectors[:,1][0] , e_vectors[:,1][1], color=['g'], scale=8)
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.show()


#%%(d) Select the correct eigenvector to form the basis of our new subspace.
#Eigenvector corresponding to higher eigenvalue is selected.
print(e_vectors[:,0])


#(e) Project the mean centered data onto this subspace and plot it.
e = np.reshape(e_vectors[:,0], (2,1))
projected_data =  np.matmul(X, e)

plt.scatter(projected_data + ox, np.zeros(20))
plt.show()


#%%(f) Transform the projected data in a way that makes it comparable to the original (non-centered data) and plot it.
#Take transpose of e
et = e.T
reconstructed_data =  np.matmul(projected_data, et)

plt.scatter(data["x"], data["y"])
plt.scatter(reconstructed_data[:,0] + ox, reconstructed_data[:,1] + oy, marker='x')
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()


#%%g) Calculate the error (w.r.t. the original data) that we make in this example by using PCA.
error = np.sum((reconstructed_data[:,0] - X[:,0])**2 + (reconstructed_data[:,1] - X[:,1])**2)
print(error)


#(h) repeat steps d)-g) but now choose the other eigenvector. Compare the results.
e = np.reshape(e_vectors[:,1], (2,1))
projected_data =  np.matmul(X, e)
plt.scatter(np.zeros(20), projected_data)
plt.show()


#Take transpose of e
et = e.T
reconstructed_data =  np.matmul(projected_data, et)
plt.scatter(data["x"], data["y"])
plt.scatter(reconstructed_data[:,0] + ox, reconstructed_data[:,1] + oy, marker='x')
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()

error = np.sum((reconstructed_data[:,0] - X[:,0])**2 + (reconstructed_data[:,1] - X[:,1])**2)
print(error)


