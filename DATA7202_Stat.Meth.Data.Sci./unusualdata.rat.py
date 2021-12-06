"""
# unusualdata.rat.py
"""
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

sns.set(style="whitegrid")

# plot pairs
ds = pd.read_csv("rat.csv",index_col=0)
pd.plotting.scatter_matrix(ds, alpha=1.0)
sns.pairplot(ds)

# fit regression
import statsmodels.formula.api as smf
results = smf.ols('y ~ BodyWt + LiverWt+Dose', data=ds).fit()
print(results.summary())

sns.residplot(ds.BodyWt, results.resid, lowess=False, color="g")
sns.residplot(ds.LiverWt, results.resid, lowess=False, color="g")
sns.residplot(ds.Dose, results.resid, lowess=False, color="g")
    
# residuals histogram
sns.distplot(results.resid)

from statsmodels.graphics.regressionplots import plot_leverage_resid2
fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(results, ax = ax)

###############################################################
# removing the third observation
ds2 = ds[ds.index !=3]
results2 = smf.ols('y ~ BodyWt + LiverWt+Dose', data=ds2).fit()
print(results2.summary())

print(results.mse_model)
print(results2.mse_model)