#empiricalcdf
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

N=1000
mu=10
X=np.random.exponential(mu, N)

x = np.linspace(0, max(X), 100)
ecdf = ECDF(X)

plt.plot(ecdf.x, ecdf.y)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Emprical CDF'])
plt.show()