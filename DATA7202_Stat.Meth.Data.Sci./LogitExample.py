from sklearn import datasets
import numpy as np
data = datasets.load_wine ()
X = data.data[:, [9 ,10]]
y = np.array(data. target ==1 , dtype =np.uint)
X = np.append(np.ones(len(X)). reshape ( -1 ,1) ,X,axis =1)


# ######################################################

import statsmodels .api as sm
model = sm.Logit (y, X)
result = model.fit()
print(result.summary())