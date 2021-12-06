"""
# modelassumption.py
"""
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing
from scipy import stats

sns.set(style="whitegrid")
ds = pd.read_csv("states.csv",index_col=False)
X = ds.loc[:, ds.columns != "Murder"]
X_num = X.loc[:, X.columns != "State"].values
y = ds.loc[:, ds.columns == "Murder"].values

regr = linear_model.LinearRegression()
regr.fit(X_num, y)
y_pred = regr.predict(X_num)

residuals = y_pred - y

# plot x vs residuals
for i in range(4):
    sns.residplot(X_num[:,i], residuals, lowess=False, color="g",label="a")
    plt.show()
   
# plot predicted vs residuals    
sns.residplot(y_pred, residuals, lowess=False, color="g")

# checking normality
sns.distplot(residuals)
plt.show()
fig = sm.qqplot(residuals[:,0])
plt.show()
