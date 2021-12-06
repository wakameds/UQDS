#%% Q1
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import os
os.chdir("/Users/wakayama.hideki/Desktop/UQ/03_Semester3/DATA7202_Stat.Meth.Data.Sci./02_Assignment/02")
#load data
df = pd.read_csv('Hitters.csv')

#label encoder
lb_make = LabelEncoder()
df.League = lb_make.fit_transform(df.League)
df.Division = lb_make.fit_transform(df.Division)
df.NewLeague = lb_make.fit_transform(df.NewLeague)

Y = df.Salary
X = df.drop(['Salary'], axis=1)

#PC regression
def PCR(X,Y):
    """
    10-Fold CV
    :param X: explanatory variables
    :param Y: response variable
    :return: dict of loss by dimension
    """
    kf = KFold(n_splits = 10, shuffle=False)
    kf.get_n_splits(X)

    Loss = {}
    ratio = {}

    for n in range(1, len(X.columns) + 1):
        pca = PCA(n_components=n)
        l = []

        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            #Fit PCA linear regression model
            x_train_pca = pca.fit_transform(x_train)
            lin_reg = LinearRegression().fit(x_train_pca, y_train)

            #prediction
            x_test_pca = pca.transform(x_test)
            y_pred = lin_reg.predict(x_test_pca)

            loss = np.mean(np.power((y_pred-y_test), 2))
            l.append(loss)
        Loss[n] = np.mean(l)
    return Loss

result = PCR(X,Y)

plt.plot(list(result.keys()), list(result.values()), '-')
plt.scatter(list(result.keys()), list(result.values()), color='blue')
plt.xticks(np.arange(0,20,1))
plt.xlabel('components')
plt.ylabel('ratio of the variance')
plt.show()

#%% b
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

lasso = linear_model.Lasso(fit_intercept=True,  max_iter=1000)
alphas = np.linspace(1, 150, 300)

scores = list()
scores_std = list()

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = -cross_val_score(lasso, X, Y, scoring='neg_mean_squared_error', cv=10)
    scores.append(np.mean(this_scores))

plt.plot(alphas, scores)
plt.xlabel('lambda')
plt.ylabel('mse')
plt.show()

#Lasso regression
def Lasso(X,Y):
    """
    10-Fold CV with Lasso method
    :param X: explanatory variables
    :param Y: response variable
    :return: dict of loss by lambda
    """
    kf = KFold(n_splits = 10, random_state=1, shuffle=True)
    kf.get_n_splits(X)
    l = []

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        #Fit linear regression model
        lasso = linear_model.Lasso(fit_intercept=True, alpha=116.65)
        lin_reg_lasso = lasso.fit(x_train, y_train)

        #prediction
        y_pred = lin_reg_lasso.predict(x_test)
        loss = np.mean(np.power((y_pred-y_test), 2))
        l.append(loss)
    print(lasso.coef_, lasso.intercept_)
    return np.mean(l)

result = Lasso(X,Y)
# 14.80: 143.698
# 116.65: 88.058



#%%Q2
import pandas as pd
import statsmodels.api as sm

df2 = pd.read_csv('ships.csv')
Y = df2['damage']
X = df2.drop(['damage'], axis=1)

model = sm.GLM(Y, X, family=sm.families.Poisson())
result = model.fit()
print(result.summary())

#%%Q3
#(a)
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

#load data
df3 = pd.read_csv('softdrink.csv')
Y = df3['Time']
X = df3.drop(['Time'], axis=1)

X = sm.add_constant(X)
mlreg = sm.OLS(Y,X).fit()
print(mlreg.summary())

#model = linear_model.LinearRegression()
#model.fit(X, Y)
y_pred = mlreg.predict(X)
residuals = y_pred - Y

#residual standard deviation
print("residual standard deviation: ",np.std(residuals))

#p-val for overall model and each predictors
print(mlreg.pvalues)

#residual plots
sns.residplot(y_pred, residuals, lowess=False, color='b')
plt.xlabel('y_predict')
plt.ylabel('residuals')
plt.show()

#residual histogram
num_bin = 10
weights = np.ones_like(residuals)/residuals.count()
plt.hist(residuals, num_bin, weights=weights, color='green', alpha=0.7)
plt.xlabel('residuals')
plt.ylabel('frequency')
plt.show()

#residual histogtam
sns.distplot(residuals)
plt.show()

#scatter plot
import pandas as pd
pd.plotting.scatter_matrix(df3, alpha=1.0)
plt.show()

#residuals by distance
X_dist = X.loc[:, X.columns == 'Distance'].values
sns.residplot(X_dist, residuals, lowess=False, color='g')
plt.xlabel('Distance')
plt.ylabel('Residual')
plt.show()

#qqplot
fig=sm.qqplot(residuals)
plt.show()

#find leverage point in "distance"
from statsmodels.graphics.regressionplots import plot_leverage_resid2
fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(mlreg, ax = ax)
plt.show()

#%%5
import numpy as np
import matplotlib.pyplot as plt
N = 1000
burn_in = 200
x = np.zeros(N+burn_in)
y = np.zeros(N+burn_in)
x[0] = 1
y[0] = 1

for i in range(1, burn_in+N):
    u = np.random.rand()
    v = np.random.rand()
    x[i] = -np.log(1-u)/(y[i-1]+1)
    y[i] = -np.log(1-v)/(x[i]+1)

f = np.exp(-(x*y+x+y))

plt.plot(x[burn_in:], y[burn_in:], '.', alpha=0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.hist(f, 30)
plt.show()
