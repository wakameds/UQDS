#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.api as sm

def CreateDataset(fName, size, seed):
    np.random.seed(seed)
    
    N = size
    
    X1 = np.random.rand(N)
    X2 = np.random.rand(N)
    X3 = np.random.rand(N)
    
    Y = np.zeros((N,))
    # logic      
    for i in range(N):
        if(np.random.rand()<0.5):
            noise = 2*np.random.rand()
        else:
            noise = -2*np.random.rand()
        
        Y[i] = 0.5*X1[i] + 3*X2[i] + 5*X3[i] + 5*X2[i]*X3[i] + 2*X1[i]*X2[i]*X3[i]  + noise
              
  
            
    data = np.array([X1, X2, X3 ,Y]).T
    df = pd.DataFrame(data,columns=['radio','tv','internet','sales'])
    #df.to_csv(fName, index=False)
    return df

if __name__ == "__main__":

    df_train = CreateDataset("synthetic_sales_train.csv", 1000, 1)  
    df_test = CreateDataset("synthetic_sales_test.csv", 1000, 2)  
    
    X_train = df_train[['radio','tv','internet']]
    y_train = df_train[['sales']].values.reshape(-1,) 
    
    lin_reg = sm.OLS(y_train, X_train).fit() 
    print(lin_reg.summary())
    
    X_test = df_test[['radio','tv','internet']]
    y_test = df_test[['sales']].values.reshape(-1,)
    
    y_hat = lin_reg.predict(X_test)
    print("regression mean squared error: ", np.mean(np.power((y_test- y_hat).values,2)))
    
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(500, random_state=0)
    reg.fit(X_train, y_train)
    y_hat = reg.predict(X_test)
    print("random forest loss: ", np.mean(np.power((y_test- y_hat),2)))
     
    print("***********************************************************")
    y_hat = lin_reg.predict(X_train)
    print("regression (training) mean squared error: ", np.mean(np.power((y_train-y_hat).values,2)))
    y_hat = reg.predict(X_train)
    print("random forest loss: ", np.mean(np.power((y_train- y_hat),2)))
    print("***********************************************************")  
    
    
    
    
    
    