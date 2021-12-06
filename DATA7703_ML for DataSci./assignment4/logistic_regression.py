import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn import linear_model 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self):
        pass

    def fit(self, X, y, lr=0.1, momentum=0, niter=100):
        '''
        Train a multiclass logistic regression model on the given training set.
        Parameters
        ----------
        X: training examples, represented as an input array of shape (n_sample,
           n_features).
        y: labels of training examples, represented as an array of shape
           (n_sample,) containing the classes for the input examples
        lr: learning rate for gradient descent
        niter: number of gradient descent updates
        momentum: the momentum constant (see assignment task sheet for an explanation)

        Returns
        -------
        self: fitted model
        '''
        self.classes_ = np.unique(y)
        self.class2int = dict((c, i) for i, c in enumerate(self.classes_))
        y = np.array([self.class2int[c] for c in y])

        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.intercept_ = np.zeros(n_classes)
        self.coef_ = np.zeros((n_classes, n_features))

        # Implement your gradient descent training code here; uncomment the code below to do "random training"
        #self.intercept_ = np.random.randn(*self.intercept_.shape)
        #self.coef_ = np.random.randn(*self.coef_.shape)

        X_tens = torch.tensor(X)
        y_tens = torch.tensor(y)
        w = torch.tensor(self.coef_, requires_grad=True)
        b = torch.tensor(self.intercept_, requires_grad=True)
        optimizer = optim.SGD([w,b], lr=lr, momentum=momentum)
        loss = nn.NLLLoss()

        #training
        for i in range(niter):
            scores = torch.matmul(X_tens, w.T) + b
            probs = F.log_softmax(scores, dim=1)
            log_loss = loss(probs, y_tens)

            optimizer.zero_grad()
            log_loss.backward()
            optimizer.step()

        self.coef_ = w.detach().numpy()
        self.intercept_ = b.detach().numpy()
        print(log_loss)

        return self

    def predict_proba(self, X):
        '''
        Predict the class distributions for given input examples.
        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).

        Returns
        -------
        y: predicted class lables, represened as an array of shape (n_sample,
           n_classes)
        '''
        #1.compute the score (score = X*w^T + b)
        scores = X@self.coef_.T + self.intercept_

        #2.subtract all scores by the max score
        maxs = np.max(scores, axis=1).reshape(-1,1)
        subtracted_scores = scores - maxs
        scores = np.exp(subtracted_scores)

        # 3.normalise the values
        sums = np.sum(scores, axis=1).reshape(-1,1)
        prob_dist = scores/sums
        return prob_dist

    def predict(self, X):
        '''
        Predict the classes for given input examples.
        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).
        Returns
        -------
        y: predicted class lables, represened as an array of shape (n_sample,)
        '''

        #1.probability distribution for each class
        probs = self.predict_proba(X)

        #2. assign class index with the max probability
        class_idxs = np.argmax(probs, axis=1)
        return self.classes_[class_idxs]


if __name__ == '__main__':
    X, y = fetch_covtype(return_X_y=True)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = linear_model.LogisticRegression(max_iter=10)
    clf.fit(X_tr, y_tr)
    print(accuracy_score(y_tr, clf.predict(X_tr)))
    print(accuracy_score(y_ts, clf.predict(X_ts)))


#%%(d)
X, y = fetch_covtype(return_X_y=True)
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)

lrs = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
iters = [1, 10, 25, 50, 75, 100,200]

clf = LogisticRegression()
for iter in iters:
    print(f'{iter}---------------------')
    clf.fit(X_tr, y_tr, lr=1e-7, niter=iter)

#lr: 1e-7, iter:inf

#%% optimized accuracy

clf = LogisticRegression()
clf.fit(X_tr, y_tr, lr=1e-7, niter=200)
print(accuracy_score(y_tr, clf.predict(X_tr)))
print(accuracy_score(y_ts, clf.predict(X_ts)))


#%%(f)
momentums = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for m in momentums:
    print(f'{m}---------')
    clf.fit(X_tr, y_tr, lr=1e-7, momentum = m, niter=200)


#%%

clf.fit(X_tr, y_tr, lr=1e-7, momentum = 0.9, niter=200)
print(accuracy_score(y_tr, clf.predict(X_tr)))
print(accuracy_score(y_ts, clf.predict(X_ts)))



#%%(g)
scaler = StandardScaler().fit(X_tr)

lrs = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
momentums = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
iters = [1, 10, 25, 50, 75, 100,200]

clf = LogisticRegression()
clf.fit(scaler.transform(X_tr), y_tr, lr=1, momentum = 0.9, niter=200)
print(accuracy_score(y_tr, clf.predict(scaler.transform(X_tr))))
print(accuracy_score(y_ts, clf.predict(scaler.transform(X_ts))))
