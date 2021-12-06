#  Copyright (c) 2015-2021, Hideki WAKAYAMA, All rights reserved.
#  File: k-NN.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 06/08/2021, 20:54

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
import random

#load data
def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, 1:-1]  # drop the first and the last features
    y = data[:, -1].astype(int)  # labels
    return X, y

#scaled X
def scale_data(X):
    scaled_data = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    return scaled_data

#split train and test sets
def train_test_split(X, y, train_size=0.7):
    num_samples = X.shape[0]  #total number of data points

    train_samples = int(num_samples * train_size)
    test_samples = num_samples - train_samples

    samples = list(range(num_samples))  # list of ids
    random.shuffle(samples)  # shuffling the ids

    #split the ids in train:test ratio
    train_ids = samples[:train_samples]
    test_ids = samples[train_samples:]

    X_train = X[train_ids]
    y_train = y[train_ids]
    X_test = X[test_ids]
    y_test = y[test_ids]

    return X_train, X_test, y_train, y_test

#accuracy score
def accuracy_score(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#calculate the Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN():
    # Initialization
    def __init__(self, K):
        self.K = K

    # Fit model to training data
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    # Predict classes of testing data
    def predict(self, X_test):
        predictions = []
        # loop over all samples in the test set
        for i in range(len(X_test)):
            # calculate the Euclidean distance between test sample and all train samples
            dist = np.array([euclidean_distance(X_test[i], x) for x in self.X_train])
            # sort the distances and return the indices of K neighbors
            sorted_dist = dist.argsort()[:self.K]
            # dictionary to store the count of neighbors according to their labels
            neighbors = {}

            # find the class for each neighbor (label,count)
            for idx in sorted_dist:
                if self.Y_train[idx] in neighbors:
                    neighbors[self.Y_train[idx]] += 1
                else:
                    neighbors[self.Y_train[idx]] = 1

            # sort the dictionary in descending order based on the label count values
            sorted_neighbors = sorted(neighbors.items(), key=operator.itemgetter(1), reverse=True)
            # append the predicted class label to the list
            predictions.append(sorted_neighbors[0][0])

        return predictions


#Perform k-NN
X, y = load_data('glass.data')
scaled_data = scale_data(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

#define a range of K to test
K_list = list(np.arange(1, int(np.sqrt(len(X))), 1))

#list to store accuracy scores
scores = []

#perform k-NN
for k in K_list:
    #train the model with the train set
    model = KNN(K=k)
    model.fit(X_train, y_train)

    #get the predictions for the test set
    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)
    scores.append(score)

#plot the classification accuracy as a function of K
plt.plot(K_list,scores)
plt.xticks(np.arange(0, max(K_list), step=5))
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.show()

best_k = K_list[scores.index(max(scores))]
print(f'best-k:{best_k}, AC:{scores[best_k]:.2f}%')