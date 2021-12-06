############################################################
# Shrinkage
#########################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1


n_points = 80 # points
X, Y =  make_friedman1(n_samples=n_points, n_features=5, 
                       noise=0.5, random_state=100)

linreg = linear_model.LinearRegression()
linreg.fit(X,Y)
linreg.intercept_
linreg.coef_


# add some redundant features
X_r = np.random.uniform(0,1,(n_points,14))
df_r = pd.DataFrame(X_r)


X = pd.DataFrame(X)
X =  pd.concat([X,df_r], axis=1)

 
scores = list()
scores_std = list()

lasso = linear_model.Lasso(fit_intercept=True)
alphas = np.logspace(-4, -.5, 30)

for alpha in alphas:
     lasso.alpha = alpha
     this_scores = -cross_val_score(lasso, X, Y,scoring='neg_mean_squared_error',cv=5)
     scores.append(np.mean(this_scores))
     scores_std.append(np.std(this_scores))
    

plt.plot(alphas, scores)
# plot error lines showing +/- std. errors of the scores
plt.plot(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)),
              'b--')
plt.plot(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)),
              'b--')
plt.ylabel('CV score (mean squared error)')
plt.xlabel('lambda')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.show()

lasso = linear_model.Lasso(fit_intercept=True,alpha=0.05)
lasso.fit(X,Y)
lasso.coef_
lasso.intercept_


