#%% Q1
import numpy as np
import pandas as pd
# generate data
def generatedata(size, seed):
    np.random.seed(seed)
    N = size
    X1 = np.random.rand(N)
    X2 = np.random.uniform(0, np.ones(N) - X1, N)
    X3 = np.ones(N) - X1 - X2
    W = np.random.rand(N)

    # 1(b) express Y and create data frame
    Y = np.zeros((N,))
    for i in range(N):
        Y[i] = 0.5 * X1[i] + 3 * X2[i] + 5 * X3[i] + 5 * X2[i] * X3[i] + 2 * X1[i] * X2[i] * X3[i] + W[i]

    data = np.array([X1, X2, X3, Y]).T
    df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'Y'])
    return df

#generate test and train data set
train_data = generatedata(1000, 1)
test_data = generatedata(1000, 2)

x_train = train_data[['X1','X2','X3']]
y_train = train_data[['Y']]

x_test = test_data[['X1','X2','X3']]
y_test = test_data[['Y']].values.reshape(-1, )

#linear regression yhat = B1*X1 + B2*X2 + B3*X3 + W
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

lin_reg = sm.OLS(y_train, x_train).fit()
print(lin_reg.summary())

#Prediction linear regression model
y_hat_lr = lin_reg.predict(x_test)
print('Regression MSE: ', np.mean(np.power((y_test - y_hat_lr).values, 2)))

#Prediction RF model
rf_reg = RandomForestRegressor(500, random_state=0)
rf_reg.fit(x_train, y_train)
y_hat_rf = rf_reg.predict(x_test)
print('RF Loss:', np.mean(np.power((y_test - y_hat_rf), 2)))

#%%Q5
import numpy as np
import statsmodels.api as sm

x1 = np.array([0, 1, 2, 3, 4])
y = np.array(([1, 2, 3, 2, 1]))

# Fit and summarize OLS model
# Model1
mod1 = sm.OLS(y,x1)
res = mod1.fit()
print(res.summary())

# Model2
X = np.vstack((np.ones(len(x1)), x1)).T
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print("beta hat is ", beta_hat)

#error loss
loss_mod1 = {}
loss_mod2 = {}

y_mod1 = res.predict(x1)
y_mod2 = X @ beta_hat

loss_mod1[2] = np.mean(np.power(y - y_mod1,2))
loss_mod1[1] = np.mean(np.abs(y - y_mod1))
loss_mod1[1.5] = np.mean(np.power(np.abs(y - y_mod1),1.5))
print("model1 loss is ", loss_mod1)

loss_mod2[2] = np.mean(np.power(y - y_mod2, 2))
loss_mod2[1] = np.mean(np.abs(y - y_mod2))
loss_mod2[1.5] = np.mean(np.power(np.abs(y - y_mod2),1.5))
print("model2 loss is ", loss_mod2)

#plot
import matplotlib.pyplot as plt
xx = range(0,5)
yy = range(0,4)
plt.plot(x1, y, '.', markersize = 8)
plt.plot(x1, y_mod1, '--', linewidth = 3)
plt.plot(x1, y_mod2, 'k--', linewidth = 3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data point', 'model1', 'model2'])
plt.show()

#%% Q6(a) load data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None
df = pd.read_csv('Hitters.csv')

#confirm null
df.isnull().sum()

#replace category with number
lb_make = LabelEncoder()
df.League = lb_make.fit_transform(df.League)
df.Division = lb_make.fit_transform(df.Division)
df.NewLeague = lb_make.fit_transform(df.NewLeague)

#check types
df.dtypes

#Q6(c) 10-Fold Cross validation
from sklearn.model_selection import KFold
import statsmodels.api as sm
import numpy as np

#Data preparation
#Separate dataset into explanatory variables and response variable
Y = df.Salary
X = df.drop(['Salary'], axis=1)

#Cross validation
def Validation(X,Y):
    kf = KFold(n_splits = 10, random_state=1, shuffle=True)
    kf.get_n_splits(X)
    err = []

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        #Fit linear regression model
        lin_reg = sm.OLS(y_train, X_train).fit()
        y_pred = lin_reg.predict(X_test)

        #Predict
        loss = np.mean(np.power((y_pred-y_test).values, 2))
        #print(loss)
        err.append(loss)
        print(lin_reg.summary())
    return np.mean(err)

Validation(X,Y)

#%% Q7
import numpy as np
from math import *

#set parameters
a = 1
b = 2
c = 3
N = 10000

ell_hat = np.zeros(N)
np.random.seed(123)
X = np.random.rand(N)

#calculate ell hat
for i in range(0,N):
    Y = 1/(a*X[i]**2 + b*X[i] + c)
    ell_hat[i] = Y

mean_ell = np.mean(ell_hat)
std_ell = np.std(ell_hat)

print('mean=', mean_ell, 'CI=(', round(mean_ell-1.96*std_ell/sqrt(N),4),
      ',',round(mean_ell+1.96*std_ell/sqrt(N),4),')')

#calculate true ell
true_ell = (2/sqrt(4*a*c - b**2))*np.arctan((2*a*1 + b)/sqrt(4*a*c - b**2)) -\
     (2/sqrt(4*a*c - b**2))*np.arctan((2*a*0 + b)/sqrt(4*a*c - b**2))

print('true ell is ', true_ell)