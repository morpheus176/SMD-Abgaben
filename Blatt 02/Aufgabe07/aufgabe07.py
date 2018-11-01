import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.stats import kde

mu_x = 4
mu_y = 2
sigma_x = 3.5
sigma_y = 1.5
covariance = 4.2


# a) Korrelationskoeffizient
rho = covariance/(sigma_x*sigma_y)
print(rho)


# c) Ellipse
np.random.seed(100)
mean = [mu_x, mu_y]
cov = [[sigma_x**2, covariance], [covariance, sigma_y**2]]

x, y = np.random.multivariate_normal(mean, cov, 10000).T        # Ziehen der Normalverteilung

w, v = LA.eig(cov)              # Bestimmung der Eigenwerte und -vektoren der Covarianz-Matrix
ind_w_max = w.argmax()
v_max = v[:, ind_w_max]         # Eigenvektor in Richtung der großen Hauptachse
ind_w_min = w.argmin()
v_min = v[:, ind_w_min]         # Eigenvektor in Rochtung der kleinen Hauptachse

theta = np.arctan(v_max[1]/v_max[0])        # Bestimmung des Winkels zwischen der großen Hauptachse und der x-Achse

chisquare_val = 1.87                   # Die Wahrscheinlich 1/e nach der Chi^2-Verteilung
theta_grid = np.linspace(0, 2*np.pi)
phi = -theta
X0 = mean[0]
Y0 = mean[1]
a = np.sqrt(chisquare_val*max(w))       # Länge der großen Hauptachse
b = np.sqrt(chisquare_val*min(w))       # Länge der kleinen Hauptachse


ellipse_x_r  = a * np.cos(theta_grid)   # Umrechnung in kartesische Koordinaten
ellipse_y_r  = b * np.sin(theta_grid)   #


R = np.array([[np.cos(phi),-np.sin(phi)], [np.sin(phi), np.cos(phi)]])      # Berechnung der Rotationsmatrix

r_ellipse = np.dot(np.array((ellipse_x_r, ellipse_y_r)).T, R)           # Drehung der Ellipse mithilfe der Rotationsmatrix



plt.plot(r_ellipse[:,0] + X0, r_ellipse[:,1] + Y0, color='lime', linestyle='-', label=r'$\frac{1}{\sqrt{e}}$ - Ellipse')                    # Plotten der Ellipse
plt.errorbar(mu_x, mu_y, xerr=sigma_x, yerr=sigma_y, capsize=5, color='cyan', marker='o', label=r'Mittelwert mit Standardabweichung')       # Plotten der Abweichungen


nbins=100                                                                                   # Plotten einer Heatmap für die gezogene Gaußverteilung
k = kde.gaussian_kde([x,y])
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='inferno')
plt.colorbar()

plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('ellipse.pdf')
