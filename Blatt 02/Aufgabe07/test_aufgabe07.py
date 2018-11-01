import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

mu_x = 4
mu_y = 2
sigma_x = 3.5
sigma_y = 1.5
cov = 4.2

np.random.seed(100)
mean = [mu_x, mu_y]
covariance = [[sigma_x**2, cov], [cov, sigma_y**2]]

x, y = np.random.multivariate_normal(mean, covariance, 5000).T

nbins = 20
k = kde.gaussian_kde(x,y)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

#plt.hist2d(x,y, cmap='inferno', bins=100)
plt.colorbar()
plt.tight_layout()

plt.show()
