import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
from scipy import linalg

X, y = make_blobs(n_samples=1000, centers=2, n_features=4,
    random_state=0)


plt.scatter(X[:,0],X[:,1], alpha=0.7)
plt.xlabel('1. Dimension')
plt.ylabel('4. Dimension')
plt.savefig('Scatterplot.pdf')
plt.clf()

#zentrieren der Werte um den Nullpunkt
mue = np.array([np.mean(X[:,0]), np.mean(X[:,1]),
                np.mean(X[:,2]), np.mean(X[:,3])])
X_zentriert = X-mue

pca = PCA(n_components=4)
pca.fit(X_zentriert)
X_pca = pca.transform(X_zentriert)


#Eigenwerte und Eigenvektoren der Kovarianzmatrix
Cov = pca.get_covariance()
print(Cov)
Eigenwerte, Eigenvektoren = linalg.eig(Cov)
print('unsortiert')
print(Eigenwerte)
#print(Eigenvektoren)


plt.scatter(X_pca[:,0],X_pca[:,1], alpha=0.7)
plt.xlabel('1. Dimension')
plt.ylabel('4. Dimension')
plt.savefig('Scatterplot_pca.pdf')
plt.clf()


plt.subplot(2, 2, 1)
plt.hist(X[:,0],color='r', density=True, bins=30, label='1. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('1. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 2)
plt.hist(X[:,1],color='y', density=True, bins=30, label='2. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('2. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 3)
plt.hist(X[:,2],color='b', density=True, bins=30, label='3. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('3. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 4)
plt.hist(X[:,3],color='g', density=True, bins=30, label='4. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('4. Dimension')
plt.savefig('Histogramm_vorher.pdf')
plt.tight_layout()
plt.clf()


plt.subplot(2, 2, 1)
plt.hist(X_pca[:,0],color='r', density=True, bins=30, label='1. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('1. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 2)
plt.hist(X_pca[:,1],color='y', density=True, bins=30, label='2. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('2. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 3)
plt.hist(X_pca[:,2],color='b', density=True, bins=30, label='3. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('3. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 4)
plt.hist(X_pca[:,3],color='g', density=True, bins=30, label='4. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('4. Dimension')
plt.savefig('Histogramm.pdf')
plt.tight_layout()
plt.clf()
