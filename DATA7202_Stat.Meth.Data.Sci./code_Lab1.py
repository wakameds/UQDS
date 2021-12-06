#%% Q1
from sklearn.datasets import make_regression, make_classification
import numpy as np
import pandas as pd

X, y = make_regression(100, 5)
Xc1, Yc1 = make_classification(100,5)

#%% Q2 (a)1
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_friedman1
from sklearn.model_selection import train_test_split
import os
os.chdir("/Users/wakayama.hideki/Desktop/UQ/03_Semester3/DATA7202_Stat.Meth.Data.Sci./03_Labs/Week2")

#%% Cross validation
X, Y = make_friedman1(n_samples = 10000, n_features=10, noise=0.5, random_state=10)
# split into train and tmp
X_train, X_tmp, y_train, y_tmp = train_test_split(X, Y, test_size=0.5, random_state=42)
# split tmp into validate and test dataset
X_validate, X_test, y_validate, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=43)

# %pwd

#%%
X_train = pd.read_csv("Xtrain.csv").values # .values returns the df
y_train = pd.read_csv("Ytrain.csv").values

X_validate = pd.read_csv("Xvalidate.csv").values
y_validate = pd.read_csv("Yvalidate.csv").values

X_test = pd.read_csv("Xtest.csv").values
y_test = pd.read_csv("Ytest.csv").values

#%%(b)(c) k-Nearest Neighbor Regression
from sklearn import neighbors

n_arr = range(1, 51)
scores = []
for k in n_arr:
    knn = neighbors.KNeighborsRegressor(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_validate)
    err = np.mean(np.power(y_pred - y_validate,2))
    scores.append(err)
    print(k, ' - ', err)
    
#%%(d) Plot validation error
plt.plot(n_arr, scores)
plt.ylabel('CV score (mean square error)')
plt.xlabel('k - # of neighbours')
plt.axhline(np.max(scores),
            linestyle='--',
            color = '0.5')
plt.show()

#%%(e) test data set
kstar_val = min(scores)
k_star = np.argmin(scores)+1
knn = neighbors.KNeighborsRegressor(k_star)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
err = np.mean(np.power(y_pred - y_test, 2))

print('The generalization error is  ', err)


#%% Q3 (a)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, zero_one_loss
import pandas as pd
import numpy as np

df = pd.read_csv('stock.csv', index_col=0)
df.Direction = df.Direction.astype('category')
df["DirectionNum"] = df.Year
df.loc[df["Direction"] == 'Up', 'DirectionNum'] = 1
df.loc[df["Direction"] == 'Down', 'DirectionNum'] = 0
print(df.dtypes)

#%% (b)
X = df.drop(['Direction', 'DirectionNum'], axis=1)
Y = df.DirectionNum

model = LogisticRegression()
model.fit(X, Y)
y_pred = model.predict(X)

print('Misclassification %= ', zero_one_loss(Y, y_pred))
print('Confusion Matrix: \n'), confusion_matrix(Y, y_pred)

#%% (c)
X = X.drop(['Today'], axis=1)
model = LogisticRegression()
model.fit(X, Y)
y_pred = model.predict(X)

print('Misclassification %= ', zero_one_loss(Y, y_pred))
print('Confusion Matrix: \n'), confusion_matrix(Y, y_pred)

#%%(d)
X_trd = X[X.Year < 2005]
Y_trd = Y[X.Year < 2005]

X_test = X[X.Year == 2005]
Y_test = Y[X.Year == 2005]

model = LogisticRegression()
model.fit(X_trd, Y_trd)
y_pred = model.predict(X_test)

print('Misclassification  %', zero_one_loss(Y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(Y_test, y_pred))

#%%(e)
X_new = X[['Lag5', 'Lag4','Lag3', 'Volume']]
X_tre = X_new[X.Year < 2005]
Y_tre = Y[X.Year < 2005]

X_test = X_new[X.Year == 2005]
Y_test = Y[X.Year == 2005]

model = LogisticRegression()
model.fit(X_tre, Y_tre)
y_pred = model.predict(X_test)
print('Misclassification  %', zero_one_loss(Y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(Y_test, y_pred))


#%% Q4
import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate # Cross-validation
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir("/Users/wakayama.hideki/Desktop/UQ/03_Semester3/DATA7202_Stat.Meth.Data.Sci./03_Labs/Week2")

#%% (a) Load and explore the data (crx.data.csv).
# Explore the frequency of += instances, and find missing values.
data = pd.read_csv('crx.data.csv')

def LoadData():
    pd.options.mode.chained_assignment = None
    # Load data
    df = pd.read_csv('crx.data.csv')
    df.dtypes

    #find and remove missing values
    for cname in df.columns:
        if (df[cname].dtype == 'object'):
            df[cname][df[cname] == '?'] = np.nan

    df = df.dropna()
    return  df

#%% (b) Prepare the data for analysis. The attribute information is as follows.
"""""
A1: b, a.
A2: continuous.
A3: continuous.
A4: u, y, l, t.
A5: g, p, gg.
A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
A7: v, h, bb, j, n, z, dd, ff, o.
A8: continuous.
A9: t, f.
A10: t, f.
A11: continuous.
A12: t, f.
A13: g, p, s.
A14: continuous.
A15: continuous.
A16: +, - (
"""""

def PrepareData():
    pd.options.mode.chained_assignment = None
    # load data
    df = LoadData()

    # Process contribution data:
    df.A2 = df.A2.astype('float64')
    df.A3 = df.A3.astype('float64')
    df.A8 = df.A3.astype('float64')
    df.A11 = df.A11.astype('float64')
    df.A14 = df.A14.astype('float64')
    df.A15 = df.A15.astype('float64')

    # Process categorical data
    lb_make = LabelEncoder()
    df.A1 = lb_make.fit_transform(df.A1)
    df.A4 = lb_make.fit_transform(df.A4)
    df.A5 = lb_make.fit_transform(df.A5)
    df.A6 = lb_make.fit_transform(df.A6)
    df.A7 = lb_make.fit_transform(df.A7)
    df.A9 = lb_make.fit_transform(df.A9)
    df.A10 = lb_make.fit_transform(df.A10)
    df.A12 = lb_make.fit_transform(df.A12)
    df.A13 = lb_make.fit_transform(df.A13)
    df.A16 = lb_make.fit_transform(df.A16)

    df.dtypes

    # get X, Y
    Y = df.A16
    X = df.drop(["A16"], axis=1)
    return X, Y

#%% (c) In this exercise, we will consider several classification algorithms and test
# their performance (zero-one loss), via 10-fold cross validation.

# i write a function that takes 3 parameters: X, Y, and a model, and returns the 10-hold
# cross validation zero-one loss estimator
def Validate(X,Y,model):
    kf = KFold(n_splits = 10, random_state = 0)
    kf.get_n_splits(X)
    zero_one_err = []

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Fit the model
        model.fit(X_train,y_train)

        # Predict
        y_pred = model.predict(X_test)
        loss = zero_one_loss(y_test, y_pred)
        zero_one_err.append(loss)
    return np.mean(zero_one_err)

# ii Write a function that implements several classifiers (Multilayer perceptron, k-N-Classifier,
# SVM, RF, and Logistic regression). The function will receive X and Y return the 10-hold cross validation
# zero-one loss for all classifier
def TestModels(X,Y):
    # Multilayer perceptron
    model = MLPClassifier(max_iter=2000)
    err = Validate(X,Y,model)
    print("------------------------------------------------")
    print("NN classifier cross validation error = ", np.mean(err))
    print("------------------------------------------------")

    # k-Neighbours classifier
    model = KNeighborsClassifier(n_neighbors=5)
    err = Validate(X,Y,model)
    print("------------------------------------------------")
    print("KNN classifier cross validation error = ", np.mean(err))
    print("------------------------------------------------")

    # Support vector machine
    model = SVC()
    err = Validate(X,Y,model)
    print("------------------------------------------------")
    print("SVC classifier cross validation error = ", np.mean(err))
    print("------------------------------------------------")

    # Random forest
    model = RandomForestClassifier(n_estimators=500, random_state=0)
    err = Validate(X,Y,model)
    print("------------------------------------------------")
    print("RF classifier cross validation error = ", err)
    print("------------------------------------------------")

    # Logistics regression
    model = LogisticRegression()
    err = Validate(X,Y,model)
    print("------------------------------------------------")
    print("Logistic regression cross validation error = ", err)
    print("------------------------------------------------")

#%% (d) Use the above functions to identify the best classifier.
X,Y = PrepareData()
print("cleaned data only")
TestModels(X,Y)

# Random forest was the best one!

#%% (e) Scale the data and repeat the classifier evaluations. Identify the best classifier.
# scale
print("scaled data")
X_scaled = scale(X)
TestModels(pd.DataFrame(X_scaled),Y)

#%% (f) Use Principal Component Regression (PCR) and repeat the classifier evaluations. Identify the best
# classifier.
print("PCR with 12 components")

ncomp = 12 # change the number of components
pca = PCA(n_components = ncomp)
X_pca = pca.fit_transform(X)
TestModels(pd.DataFrame(X_pca),Y)

#%% (g) What is your conclusion?
#Compare the obtained results with the paper Simplifying decision trees.pdf.

#%% Q5
import numpy as np
from math import *

def f(x, y):
    if(x<0.5 and y<0.5):
        return 1
    else:
        return 0

N =1000
ell = np.zeros(N)

for i in range(0, N):
    X = np.random.uniform(0,1)
    Y = np.random.uniform(0,1)

    ell[i] = f(X, Y)

ell_mean = np.mean(ell)
ell_std = np.std(ell)

print('mean=', ell_mean, 'CI=(', round(ell_mean-1.96*ell_std/sqrt(N),4),
      ',',round(ell_mean+1.96*ell_std/sqrt(N),4),')')

