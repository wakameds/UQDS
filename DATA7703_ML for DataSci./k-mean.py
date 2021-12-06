import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as metric
from scipy.spatial.distance import cdist
import random

#load data
data = pd.read_csv('simple.txt', sep='\s+')
#convert to numpy array
X = data.to_numpy(dtype = 'float')

#plot the data points
plt.scatter(X[:, 0], X[:,1], marker='.')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()

print(X.shape, X.dtype, len(X))
print(X[0,:])


#function
#1.find euclidean distance
def euclidean_dist(x1,x2):
    """
    function to compute L2 norm
    :param x1: instance A
    :param x2: instance B
    :return: distance
    """
    return metric.euclidean(x1, x2)

#2. plot the different points and new centroid position with different colors
def plot(data, k, index, centroids):
    """
    function to figure plot for k_mean
    :param data: dataset
    :param k: the parameter with k
    :param index: class labels for the data
    :param centroids: centroids points
    :return: scatter plot
    """
    input = []

    for i in range(len(index)):
        for j in index[i]:
            input.append(int(j))

    colors = 10*['g','r','c','b','k']

    j=0
    for i in input:
        #color is identified by class label
        plt.scatter(X[j,0], X[j,1], marker='.', color=colors[i], s=50, linewidth=5)
        j += 1

    #New centroids
    for centroid in range(len(centroids)):
        plt.scatter(centroids[centroid][0], centroids[centroid][1], marker='x', color ='k',
                    label = 'Centroids %d'%centroid,
                    s =130, linewidths=5)
        plt.legend()


#K-means
class k_means():
    """
    class for k_kean
    """
    def __init__(self, k, data, num, centroid_init = None):
        """
        :param k: k
        :param data: dataset
        :param num: max iteration of k-mean procedure
        :param centroid_init: centroid
        """
        self.k = k
        self.data = data
        self.num = num
        self.centroid_init = centroid_init

    def initialise_centroid(self, centroid_init, k, data):
        """
        function to decide the initial centroids by k
        :param centroid_init(str): 'random' or 'other'
        :param k(int): the parameter with k
        :param data(array): dataset
        :return: initial centroid sample
        """
        #1. random case
        if self.centroid_init == 'random':
            #shuffle data and decide k points as initial centroid points
           initial_centroids = np.random.permutation(data.shape[0])[:self.k]
           self.centroids = data[initial_centroids]
        #2. not shuffle case. Decide the centroid pints from the top row
        elif(self.centroid_init == 'other'):
            self.centroids = data[:k]

        return self.centroids

    def fit(self, data):
        #make matrix for class
        sample_num = np.shape(data)[0]
        cluster_assignments = np.mat(np.zeros((sample_num,2)))

        #set initial centroids
        cents = self.initialise_centroid(self.centroid_init, self.k, data)

        #preserve original centroids
        cents_orig = cents.copy()
        changed = True
        num_iter = 0

        #iterate update until num_iter reachs set num count
        while changed and num_iter < self.num:
            changed = False
            #for each row in the dataset
            for i in range(sample_num):
                # Track minimum distance and vector index of associated cluster
                min_dist = np.inf
                min_index = -1
                #calculate distance
                for j in range(self.k):
                    dist_ji = euclidean_dist(cents[j,:], data[i,:])
                    if dist_ji < min_dist:
                        min_dist = dist_ji
                        min_index = j
                    #check if cluster assignment of instance has changed
                    if cluster_assignments[i,0] != min_index:
                        changed = True

                #Assign instance to appropriate cluster
                cluster_assignments[i, :] = min_index, min_dist**2

            #Update centroid location by computing mean
            for cent in range(self.k):
                points = data[np.nonzero(cluster_assignments[:,0].A==cent)[0]]
                cents[cent, :] = np.mean(points, axis=0)

            #Count iterations
            num_iter += 1
            print('no of iter:', num_iter)

        #Return
        return cents, cluster_assignments, num_iter


#Perform k-means clustering with centroids initialize='random'
kmeans = k_means(k=3, data= X, num=10, centroid_init='random')
centroids, cluster_assignments, iters = kmeans.fit(X)
index = cluster_assignments[:, 0]
k = 3

#plot the clusters and their centroids
plot(data, k, index, centroids)
plt.show()



#Perform k-means clustering with other iteration'
kmeans = k_means(k=3, data= X, num= 2, centroid_init='other')
centroids, cluster_assignments, iters = kmeans.fit(X)
index = cluster_assignments[:, 0]
k = 3

#plot the clusters and their centroids
plot(data, k, index, centroids)
plt.show()


#plot
costs = []
for i in range(7):
    kmeans = k_means(k=3, data=X, num=i, centroid_init= 'other')  # try random initalization here
    centroids, cluster_assignments, iters = kmeans.fit(X)
    distance = cluster_assignments[:, 1]  ## This has the distance from their respective centroides for evaluation purposes
    cost = sum(distance)
    cost = np.array(cost)
    cost = cost.item()
    costs.append(cost)

x = np.arange(7)
plt.plot(x, costs)
plt.title("Plot")
plt.xlabel("iterations")
plt.ylabel("distance")
plt.show()

#%%

#updata the first distance
data_fish = pd.read_csv("fish.txt",  sep="\s+")

X = data_fish.iloc[:, 1:]
y = data_fish.iloc[:, 0]

#convert to numpy array
X = X.to_numpy(dtype='float')

#check the shape of the dataset
print(X.shape)
print(y.shape)

#check the data type of the dataset
print(X.dtype)

#check the no of samples in the dataset
print(len(X))

#check the magnitude of the data elements in the list row
print(X[0, :])


#Create k-means instance : kmeans for fish dataset
kmeans = k_means(k=4, data = X, num=7, centroid_init='random')
centroids, cluster_assignments, iters = kmeans.fit(X)
index = cluster_assignments[:,0]
k=4
plot(data, k, index, centroids)
plt.show()


#Centring and normalization of the fish dataset
X = (X - np.mean(X))/np.std(X)
#check the data type of the dataset
print(X.dtype)
#check the no of samples in the dataset
print(len(X))
#check the magnitude of the data elements in the list row
print(X[0, :])

#Create k-means instance : kmeans for fish dataset
kmeans = k_means(k=4, data = X, num=1, centroid_init='random')
centroids, cluster_assignments, iters = kmeans.fit(X)
index = cluster_assignments[:,0]
k=4
plot(data, k, index, centroids)
plt.show()

