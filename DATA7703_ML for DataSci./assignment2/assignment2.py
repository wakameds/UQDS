import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("04_Assignment/assignment2")


# Load data
X_tr = pd.read_csv('faces_train.txt', sep='\s+', header=None).T
y_tr = pd.read_csv('faces_train_labels.txt', sep='\s+', header=None)
X_test = pd.read_csv('faces_test.txt', sep='\s+', header=None).T
y_test = pd.read_csv('faces_test_labels.txt', sep='\s+', header=None)

# Task1.
# Confirm image
img = np.rot90(np.reshape(X_tr[14].values, (32, 32)), k=3)
plt.imshow(img, cmap='gray')
plt.title('confirm image')
plt.show()


# Task2.
# Find the mean face image on the training faces
mean = X_tr.mean(axis=1)

#  Mean face image
plt.imshow(np.rot90(np.reshape(mean.values, (32,32)), k=3), cmap='gray')
plt.title('mean face image')
plt.show()


# Extract the top of K components eigen face
# Perform PCA
X_centred = X_tr.sub(mean, axis='index')
cov = np.cov(X_centred)
eig_vals, eig_vectors = np.linalg.eig(cov)

#complex into float
eig_vals = np.real_if_close(eig_vals, tol=1)
eig_vectors = np.real_if_close(eig_vectors, tol=1)


# Confirm cumulative distribution of the components
n_component = 150
eig_vals = eig_vals[:n_component]
eig_vals_ratio = eig_vals/eig_vals.sum()
cumulative = np.cumsum(eig_vals_ratio)
plt.plot(np.arange(1,n_component+1), cumulative, marker='.')
plt.xlabel('component')
plt.ylabel('rate of the accumulated eigenvalues')
plt.title('cumulative distribution')
plt.show()


# Extract top 5 components for the train
dim = 5
components = eig_vectors[:, :dim]

def eigface(components, titles, row, col):
    plt.figure(figsize=(1.5 * col, 2.5*row))
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.imshow(np.rot90(np.reshape(components[:, i], (32,32)), k=3), cmap='gray')
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

titles = ['eigface %d' %(i+1) for i in range(dim)]
eigface(components, titles, 1, 5)
plt.suptitle('top5 eigenfaces by PCA')
plt.show()


# Task 3. & Task4.
# Compute k-dimensional projection of the test images onto the face space
dims = [2,4,6,20]
X_reconstruct_test = []

for dim in dims:
    components = eig_vectors[:, :dim]
    X_test_centred = X_test.sub(X_test.mean(axis=1), axis='index')
    X_transpose_test = (np.dot(components.T, X_test_centred[:][1]))

    # Reconstruct the approximated faces and display them as images again
    X_reconstruct_test.append(np.dot(components, X_transpose_test))


#show a image on the test by differet k
titles = ['k = %d' %dim for dim in dims]
plt.figure(figsize=(10, 2.5))
for i in range(len(dims)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(np.rot90(np.reshape(X_reconstruct_test[i], (32, 32)), k=3), cmap='gray')
    plt.title(titles[i], size=12)
    plt.xticks(())
    plt.yticks(())
plt.suptitle('reconstructed a face image of the test by different value of k')
plt.show()


# Task 5.
# Find the closest training image by L2 to the test image in face space
# Assign the label of the training image to the test image
class eigenface():
    def __init__(self, eigenvectors, X_tr, y_tr, X_test, y_test, dim):
        self.eigenvectors = eigenvectors
        self.X_tr = X_tr
        self.X_test = X_test
        self.y_tr = y_tr
        self.y_test = y_test
        self.dim = dim


    def train_test(self):
        components = self.eigenvectors[:, :self.dim].astype(float)
        X_transpose_tr = np.dot(components.T, self.X_tr.sub(self.X_tr.mean(axis=1), axis='index'))
        X_transpose_test = np.dot(components.T, self.X_test.sub(self.X_test.mean(axis=1), axis='index'))
        return X_transpose_tr, X_transpose_test

    def predict_label(self, x_transpose_test, X_transpose_tr):
        _, faces = X_transpose_tr.shape
        #compute L2 norm between x_test and x_train
        l2 = [np.linalg.norm(x_transpose_test - X_transpose_tr[:, face]) for face in range(faces)]
        #find closest distance and index of X_train
        min_idx = l2.index(min(l2))
        y_pred = y_tr.values[0][min_idx]
        return y_pred, min_idx

    def accuracy(self, y_pred, y_test):
        correct = 0
        row, col = y_test.shape
        for i in range(col):
            if y_pred[i] == y_test.values[0][i]:
                correct += 1
        acc = correct/col * 100
        return acc

    def incorrect_idxs(self, y_pred, y_test):
        incorrect_idx = []
        row, col = y_test.shape
        for i in range(col):
            if y_pred[i] != y_test.values[0][i]:
                incorrect_idx.append(i)
        return incorrect_idx


K = np.arange(1,30)
accs = []

for k in K:
    model = eigenface(eig_vectors, X_tr, y_tr, X_test, y_test, dim=k)
    X_transpose_tr, X_transpose_test = model.train_test()
    _, faces = X_transpose_test.shape
    y_preds = [model.predict_label(X_transpose_test[:, face], X_transpose_tr)[0] for face in range(faces)]
    accs.append(model.accuracy(y_preds, y_test))

plt.plot(K, accs, marker='.')
plt.xlabel('components')
plt.ylabel('accuracy')
plt.show()


#incorrect classified five faces in k=100
model = eigenface(eig_vectors, X_tr, y_tr, X_test, y_test, dim = 100)
X_transpose_tr, X_transpose_test = model.train_test()
_, faces = X_transpose_test.shape
y_preds = [model.predict_label(X_transpose_test[:, face], X_transpose_tr)[0] for face in range(faces)]

#index of the incorrect test image
y_test_idx = model.incorrect_idxs(y_preds, y_test)

#index of the image on training which is close to the test image
X_tr_idxs = [model.predict_label(X_transpose_test[:, face], X_transpose_tr)[1] for face in range(faces)]

#Extract the index of the training image, which close to the incorrected test image
X_tr_idxs2 = [X_tr_idxs[idx] for idx in y_test_idx]


images = []
labels = []

for idx in y_test_idx[:5]:
    labels.append('incorrect test label %d' %y_test.values[0][idx])
    images.append(X_test.values[:,idx])

for idx in X_tr_idxs2[:5]:
    labels.append('closed train label %d' %y_tr.values[0][idx])
    images.append(X_tr.values[:,idx])


plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.rot90(np.reshape(images[i], (32, 32)), k=3), cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title(labels[i], size=9)
plt.suptitle('misclassified face images')
plt.show()