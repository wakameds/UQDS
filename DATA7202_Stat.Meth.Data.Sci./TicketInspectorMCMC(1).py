"""
TicketInspectorMCMC.py
"""

import numpy as np
from scipy.stats import gamma

np.random.seed(12345)

x = 60
# simulation parameters
N = 10000
burnin = 1000
# array for theta recording
ell = np.zeros((N,2))
# initial values
t = 1
k = 1
# Main MCMC loop
for i in range(N+burnin):
    # sample k
    prob = [np.exp(-12*t), np.exp(-21*t)*(2**(x-1)), np.exp(-30.6667*t)*(3**(x-1))]
    prob = prob/np.sum(prob)
    k = np.random.choice([1,2,3], 1, p=prob)
    
    # sample t
    alpha = x+1 
    betta  = 2/k + 10*k
    t = gamma.rvs(a = alpha,scale = 1/betta)
    
    if(i>=burnin):
        ell[i-burnin][0] = k
        ell[i-burnin][1] = t

print("P(k=1 | x=60 ) = ",np.mean(ell[:,0]==1))
print("P(k=2 | x=60 ) = ",np.mean(ell[:,0]==2))
print("P(k=3 | x=60 ) = ",np.mean(ell[:,0]==3))
