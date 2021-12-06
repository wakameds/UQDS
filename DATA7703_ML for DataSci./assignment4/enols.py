import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import clone 
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

def corrupt(X, y, outlier_ratio=0.1, random_state=None):
    random = check_random_state(random_state)

    n_samples = len(y)
    n_outliers = int(outlier_ratio*n_samples)

    W = X.copy()
    z = y.copy()

    mask = np.ones(n_samples).astype(bool)
    outlier_ids = random.choice(n_samples, n_outliers)
    mask[outlier_ids] = False

    W[~mask, 4] *= 0.1

    return W, z

class ENOLS:
    def __init__(self, n_estimators=100, sample_size='auto'):
        '''
        Parameters
        ----------
        n_estimators: number of OLS models to train
        sample_size: size of random subset used to train the OLS models, default to 'auto'
            - If 'auto': use subsets of size n_features+1 during training
            - If int: use subsets of size sample_size during training
            - If float: use subsets of size ceil(n_sample*sample_size) during training
        '''

        self.n_estimators = n_estimators
        self.sample_size = sample_size

    
    def fit(self, X, y, random_state=None):
        '''
        Train ENOLS on the given training set.

        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        y: an array of shape (n_sample,) containing the values for the input examples

        Return
        ------
        self: the fitted model
        '''
        n_samples, n_features = X.shape
        if self.sample_size == 'auto':
            sample_size = n_features + 1

        elif type(self.sample_size) == int:
            sample_size = self.sample_size

        elif type(self.sample_size)==float:
            sample_size = int(np.ceil(n_samples*self.sample_size))

        # use random instead of np.random to sample random numbers below
        random = check_random_state(random_state)

        # add all the trained OLS models to this list
        self.estimators_ = []

        # write your training code below. your code should support the
        # n_estimators and sample_size hyper-parameters described in the
        # documentation for the __init__ function

        #sampling N random subsets consisting of m samples
        for i in range(self.n_estimators):
            model = LinearRegression()
            idxs = random.choice(len(X), size = sample_size, replace=False)
            X_tr, y_tr = X[idxs], y[idxs]
            #train OLS model by each subset
            model.fit(X_tr, y_tr)
            self.estimators_.append(model)
        return self
    
    def predict(self, X, method='average'):
        '''
        Parameters
        ----------
        X: an input array of shape (n_sample, n_features)
        method: 'median' or 'average', corresponding to predicting median and
            mean of the OLS models' predictions respectively.

        Returns
        -------
        y: an array of shape (n_samples,) containig the predicted values
        '''

        if method == 'average':
            y_pred_sum = np.zeros(len(X))
            for i in range(self.n_estimators):
                y_pred_sum += self.estimators_[i].predict(X)
            y_pred = y_pred_sum/self.n_estimators

        elif method == 'median':
            y_median = np.zeros((len(X), self.n_estimators))
            for i in range(self.n_estimators):
                y_pred_median = self.estimators_[i].predict(X)
                y_median[:, i] = y_pred_median
            y_pred = np.median(y_median, axis=1)

        return y_pred



if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)
    W, z = corrupt(X_tr, y_tr, outlier_ratio=0.1, random_state=42)
    
    reg = LinearRegression()
    reg.fit(X_tr, y_tr)
    print(mean_squared_error(y_ts, reg.predict(X_ts)))

    reg = LinearRegression()
    reg.fit(W, z)
    print(mean_squared_error(y_ts, reg.predict(X_ts)))


#(d) compare the three models by probability
X, y = load_boston(return_X_y=True)
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)
W, z = corrupt(X_tr, y_tr, outlier_ratio=0.1, random_state=42)

ps = np.arange(0, 0.5, 0.01)
Ws = []
zs = []

for p in ps:
    W, z = corrupt(X_tr, y_tr, outlier_ratio=p, random_state=42)
    Ws.append(W)
    zs.append(z)


OLSs =[]
Theil_Sens =[]
ENOLSs = []


for i in range(len(ps)):
    W, z = Ws[i], zs[i]
    #OLS
    reg = LinearRegression()
    reg.fit(W, z)
    OLSs.append(mean_squared_error(y_ts, reg.predict(X_ts)))

    #Theil-sen
    reg_ts = TheilSenRegressor()
    reg_ts.fit(W, z)
    Theil_Sens.append(mean_squared_error(y_ts, reg_ts.predict(X_ts)))

    #ENLOS
    reg_enlos = ENOLS()
    reg_enlos.fit(W,z, random_state=123)
    ENOLSs.append(mean_squared_error(y_ts, reg_enlos.predict(X_ts)))

plt.plot(ps, OLSs, label = 'OLS')
plt.plot(ps, Theil_Sens, label='Theil-sen')
plt.plot(ps, ENOLSs, label='ENOLS_average')
plt.xlabel('proportion')
plt.ylabel('mse')
plt.legend()
plt.show()

#%%(e) ENOLS method 'median'
ENOLSs_med =[]

for i in range(len(ps)):
    W, z = Ws[i], zs[i]
    reg_enlos = ENOLS()
    reg_enlos.fit(W,z, random_state=123)
    ENOLSs_med.append(mean_squared_error(y_ts, reg_enlos.predict(X_ts, method='median')))

plt.plot(ps, OLSs, label = 'OLS')
plt.plot(ps, Theil_Sens, label='Theil-sen')
#plt.plot(ps, ENOLSs, label='average')
plt.plot(ps, ENOLSs_med, label='ENOLS_median')
plt.xlabel('proportion')
plt.ylabel('mse')
plt.legend()
plt.show()


#%%(f) set n_estimator = 500
ENOLSs_med_500 =[]

for i in range(len(ps)):
    W, z = Ws[i], zs[i]
    reg_enlos = ENOLS(n_estimators=500)
    reg_enlos.fit(W,z, random_state=123)
    ENOLSs_med_500.append(mean_squared_error(y_ts, reg_enlos.predict(X_ts, method='median')))

plt.plot(ps, OLSs, label = 'OLS')
plt.plot(ps, Theil_Sens, label='Theil-sen')
plt.plot(ps, ENOLSs_med_500, label='ENOLS_median_500')
plt.xlabel('proportion')
plt.ylabel('mse')
plt.legend()
plt.show()

#%%(g)set samplesize=42
ENOLSs_med_500_size42 =[]

for i in range(len(ps)):
    W, z = Ws[i], zs[i]
    reg_enlos = ENOLS(n_estimators=500, sample_size=42)
    reg_enlos.fit(W,z, random_state=123)
    ENOLSs_med_500_size42.append(mean_squared_error(y_ts, reg_enlos.predict(X_ts, method='median')))

plt.plot(ps, OLSs, label = 'OLS')
plt.plot(ps, Theil_Sens, label='Theil-sen')
plt.plot(ps, ENOLSs_med_500_size42, label='ENOLS_median_500_ss42')
plt.xlabel('proportion')
plt.ylabel('mse')
plt.legend()
plt.show()

