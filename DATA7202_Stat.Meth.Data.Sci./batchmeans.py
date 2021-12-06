import numpy as np
M = 10000
K = 1000
N = 50

#sample from N(5,9)
ell = np.random.randn(M,1)*3 + 5
T = float((M-K)/N)
ell = ell[K+1:len(ell)]
batch_ell = np.zeros(N)

#%%
for i in range(N):
    startid = T*i+1
    endid = startid+T
    tmp = ell[startid:endid]
    batch_ell[i] = np.mean(tmp)