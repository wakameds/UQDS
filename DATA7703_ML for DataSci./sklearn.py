from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

#Q1 load dataset
X,y = load_boston(return_X_y = True)
print(X.shape, y.shape)

#Q2 split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape)

def evaluate(reg, X_tr, y_tr, X_ts, y_ts, N=1, seed=42):
    if seed:  # make results reproducible
        np.random.seed(seed)

    mse = []
    for i in range(N):
        reg.fit(X_tr, y_tr)
        r2_tr = reg.score(X_tr, y_tr)
        mse_tr = mean_squared_error(y_tr, reg.predict(X_tr))
        mse_ts = mean_squared_error(y_ts, reg.predict(X_ts))
        mse.append([r2_tr, mse_tr, mse_ts])
    return pd.DataFrame(mse, columns=['R2 (train)', 'MSE (train)', 'MSE (test)'])


#lr
from sklearn.linear_model import LinearRegression
boston = X_train, X_test, y_train, y_test
perf_ols = evaluate(LinearRegression(), X_train, y_train, X_test, y_test, N=3)
print(perf_ols)

#decision tree
from sklearn.tree import DecisionTreeRegressor

perf_dt = evaluate(DecisionTreeRegressor(),  X_train, y_train, X_test, y_test, N=100)
perf_dt.describe()


#(d)Bagging
from sklearn.ensemble import BaggingRegressor

perf_bagging = evaluate(BaggingRegressor(n_estimators=50), X_train, y_train, X_test, y_test, N=100)
perf_bagging.describe()


#(e)RandomForest
from sklearn.ensemble import RandomForestRegressor

perf_rf = evaluate(RandomForestRegressor(n_estimators=50), X_train, y_train, X_test, y_test, N=100)
perf_rf.describe()