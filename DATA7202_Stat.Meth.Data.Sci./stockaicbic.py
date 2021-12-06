############################################################
# Stock market direction prediction AIC/BIC
#########################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss

df = pd.read_csv("stock.csv",index_col=0)
df.Direction = df.Direction.astype('category')
df["DirectionNum"] = df.Year
df.loc[df['Direction'] =="Up", 'DirectionNum'] = 1
df.loc[df['Direction'] =="Down", 'DirectionNum'] = 0

# construct X,Y
Y = df.DirectionNum
X = df.drop(["Direction","DirectionNum","Year"],axis=1)

X1 = X.drop(["Today"],axis=1)
X2 = X.drop(["Lag5","Lag4","Lag3","Volume","Today"],axis=1)


model1 = LogisticRegression(solver='lbfgs')
model1.fit(X1,Y)

model2 = LogisticRegression(solver='lbfgs')
model2.fit(X2,Y)



def log_likelihood(X, Y, model):
   tX = X.values
   tY = Y.values
   n = len(X)
   logl = 0
   for i in range(0,n):    
       score =  np.sum((tX[i,:]*model.coef_ + model.intercept_))
       logl = logl + tY[i]*score - np.log(1 + np.exp(score))
   
   return logl

logl1 = log_likelihood(X1,Y,model1)
AIC1 = -2*logl1 +  2*X1.shape[1]   
BIC1 = -2*logl1 +  X1.shape[1] *np.log(len(X1))
    
logl2 = log_likelihood(X2,Y,model2)
AIC2 = -2*logl2 +  2*X2.shape[1]   
BIC2 = -2*logl2 +  X2.shape[1] *np.log(len(X2))

print("AIC model 1 = ", AIC1, "AIC model 2 = ", AIC2)
print("BIC model 1 = ", BIC1, "BIC model 2 = ", BIC2)




