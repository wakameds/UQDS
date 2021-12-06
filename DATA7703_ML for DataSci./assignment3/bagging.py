import sklearn.datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

#Q3
#(a)load the dataset
X, y = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, shuffle=True, random_state=34)

#%%


#(b)train a random forest with 100 decision trees with default hparams
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_train)
y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)

acc_train = regr.score(X_train, y_train)
acc_test = regr.score(X_test, y_test)
print('train R^2: {}'.format(round(acc_train, 3)))
print('test R^2: {}'.format(round(acc_test, 3)))


#(c)compute correlations and the mean
predictions_all = np.array([tree.predict(X_test) for tree in regr.estimators_])
correlation = np.corrcoef(predictions_all)
np.mean(correlation)


#(d)Visualize the change of the train and test set accuracy by different m
d = 8
accs_train = []
accs_test = []
corrs = []

for m in range(1, d+1):
    regr = RandomForestRegressor(max_features=m)
    regr.fit(X_train, y_train)
    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)
    #accuracy(R2)
    accs_train.append(regr.score(X_train, y_train))
    accs_test.append(regr.score(X_test, y_test))
    predictions_all = np.array([tree.predict(X_test) for tree in regr.estimators_])
    #correlation
    correlation = np.corrcoef(predictions_all)
    corrs.append(np.mean(correlation))

#plot
X = np.linspace(1,8,8)
plt.plot(X, accs_train, label='train', marker='.')
plt.plot(X, accs_test, label='test', marker='.')
plt.xlabel('d')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(X, corrs, marker='.')
plt.xlabel('d')
plt.ylabel('correlation')
plt.show()


#%%4
#(a)load digit dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=12, shuffle=True)


#(b) (c) (d)
from sklearn.base import clone
import numpy as np
from sklearn.metrics import zero_one_loss

class OOBaggingClassifier:
    def __init__(self, base_estimator, n_estimators=200):
        '''
        Parameters
        ----------
        base_estimator: a probabilistic classifier that implements the predict_prob function, such as DecisionTreeClassifier
        n_estimators: the maximum number of estimators allowed.
        '''
        self.base_estimator_ = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.oob_errors_ = []

    def fit(self, X, y, random_state=10):
        if random_state:
            np.random.seed(random_state)

        self.best_n = 0
        probs_oob = np.zeros((len(X), len(np.unique(y))))

        for m in range(self.n_estimators):
            estimator = clone(self.base_estimator_)

            # construct a bootstrap sample
            idxs_boot = np.random.choice(len(X), size=len(X))
            X_boot, y_boot = X[idxs_boot], y[idxs_boot]

            # extract oob samples from train set
            idxs_oob = np.array([idx for idx in range(len(X)) if idx not in idxs_boot])
            X_oob, y_oob = X[idxs_oob], y[idxs_oob]

            # train on bootstrap sample
            estimator.fit(X_boot, y_boot)

            # predict max probabilities with each OOB samples
            oob_probs = estimator.predict_proba(X_oob)

            #update oob probability
            probs_oob[idxs_oob] += oob_probs

            #compute OOB error
            oob_pred = np.argmax(probs_oob, axis=1)
            oob_error = zero_one_loss(oob_pred, y)

            # save the OOB error and the new model
            self.oob_errors_.append(oob_error)
            self.estimators_.append(estimator)

            # stop early if smoothed OOB error increases (for the purpose of
            # this problem, we don't stop training when the criterion is
            # fulfilled, but simply set self.best_n to (i+1)).
            if (self.best_n ==0) and (m>=10) and (np.mean(self.oob_errors_[-5:])>np.mean(self.oob_errors_[-10:-5])):  # replace OOB criterion with your code
                self.best_n = (m+1)


    def errors(self, X, y):
        '''
        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        y: an array of shape (n_sample,) containing the classes for the input examples
        Returns
        ------
        error_rates: an array of shape (n_estimators,), with the error_rates[i]
        being the error rate of the ensemble consisting of the first (i+1)
        models.
        '''
        error_rates = []
        scores = None
        # compute all the required error rates
        for estimator in self.estimators_:
            p = estimator.predict_proba(X)
            if scores is None:
                scores = p
            else:
                scores += p
            preds = np.argmax(scores,axis=1)
            error_rates.append(zero_one_loss(preds, y))
        return error_rates

    def predict(self, X):
        '''
        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        Returns
        ------
        y: an array of shape (n_samples,) containing the predicted classes
        '''
        probs = None
        for estimator in self.estimators_:
            p = estimator.predict_proba(X)
            if probs is None:
                probs = p
            else:
                probs += p
        return np.argmax(probs, axis=1)


#%%
#(e)
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

rstate = 200

bagging = OOBaggingClassifier(DecisionTreeClassifier(),n_estimators=200)
bagging.fit(X_train, y_train, random_state=rstate)
error_rates = bagging.errors(X_test, y_test)
m = np.arange(len(bagging.oob_errors_))

plt.plot(m, bagging.oob_errors_, label='oob')
plt.plot(m, error_rates, label='test')
plt.vlines(bagging.best_n, 0, 0.7, ls='--', colors='r')
plt.xlabel('the number of basis models')
plt.ylabel('error')
plt.legend()
plt.title('random state = {}, m = {}'.format(rstate, bagging.best_n))
plt.show()

print('The best number of the basis models: {}'.format(bagging.best_n))
