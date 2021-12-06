############################################################
# Stock market direction prediction feature selection
############################################################

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold


####################################################################


df = pd.read_csv("stock.csv",index_col=0)
df.Direction = df.Direction.astype('category')
df["DirectionNum"] = df.Year
df.loc[df['Direction'] =="Up", 'DirectionNum'] = 1
df.loc[df['Direction'] =="Down", 'DirectionNum'] = 0

# construct X,Y
Y = df.DirectionNum
X = df.drop(["Direction","DirectionNum","Today"],axis=1)


logreg = LogisticRegression(solver='lbfgs')
rfecv = RFECV(estimator=logreg, step=1, cv=KFold(5),
              scoring='accuracy')
rfecv.fit(X, Y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (# of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# we can see that Lag1 + Volume are the best features
X_new = rfecv.transform(X)