import numpy as np

#finitehorizon
N = 1000

#smaple from N(5, 9)
ell = np.random.randn(5,9,1)*3+5
mu = np.mean(ell)
S = np.std(ell)
CI = [mu-1.96*S/np.sqrt(N), mu+1.96*S/np.sqrt(N)]
print('0.95 CI for is (%f, %f)' % (CI[0], CI[1]))