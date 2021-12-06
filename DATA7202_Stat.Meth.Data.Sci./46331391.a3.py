#%% Q5
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

#create regression problem
n_points = 1000
x, y = make_friedman1(n_samples= n_points, n_features = 15, noise=1.0, random_state=100)

#split to train/test set
x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=100)

#Randomforest
R2 = list()
for m in range(1,16):
    rfc = RandomForestRegressor(n_estimators=1000, max_features=m, random_state=100)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    R2.append(r2)

R22 = dict()
for i in range(0,15):
    R22[i+1] = R2[i]

plt.plot(range(1,16), R2, '-')
plt.scatter(range(1,16), R2)
plt.xlabel('the number of predictors')
plt.ylabel('R2')
plt.show()

print('the optimized number of the predictors is ',max(R22, key=R22.get))


#%%6
from sklearn.datasets import make_blobs
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

X_train, y_train = make_blobs(n_samples=1000, n_features=10, centers=3,
                              random_state=10, cluster_std=5)

#boosting sklearn
R = [0.1, 0.3, 0.5, 0.7, 1]
B = 150
x_line = np.linspace(1,B,B)

for r in R:
    breg = GradientBoostingClassifier(learning_rate = r, n_estimators = B, random_state=100)
    breg.fit(X_train, y_train)
    y_pred = breg.predict(X_train)
    plt.plot(x_line, breg.train_score_)

plt.legend([f"\u03B3={r}" for r in R])
plt.xlabel('B')
plt.ylabel('Loss')
plt.show()

