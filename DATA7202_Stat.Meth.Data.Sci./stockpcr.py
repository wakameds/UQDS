############################################################
# Stock market direction prediction with PCA
#########################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix

df = pd.read_csv("stock.csv",index_col=0)
df.Direction = df.Direction.astype('category')
df["DirectionNum"] = df.Year
df.loc[df['Direction'] =="Up", 'DirectionNum'] = 1
df.loc[df['Direction'] =="Down", 'DirectionNum'] = 0

# construct X,Y
Y = df.DirectionNum
X = df.drop(["Direction","DirectionNum","Today"],axis=1)

# train test set
Xtr = X[X.Year<2005]
Ytr = Y[X.Year<2005]

Xtest = X[X.Year == 2005]
Ytest = Y[X.Year == 2005]

# PCA
ncomp = 4 # change the number of components
pca = PCA(n_components=ncomp)
X_pca = pca.fit_transform(Xtr)


# fit the model
model = LogisticRegression(solver='lbfgs')
model.fit(X_pca,Ytr)

# predict
x_test_pca = pca.transform(Xtest)
y_pred = model.predict(x_test_pca)

print("misclassification % = ",zero_one_loss(Ytest,y_pred))
print("Confusion matrix: \n", confusion_matrix(Ytest,y_pred))

