import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats
from scipy import linalg

P0 = np.array([np.matrix([1,1]), np.matrix([2,1]), np.matrix([1.5,2]), np.matrix([2,2]), np.matrix([2,3]), np.matrix([3,3])])
P1 = np.array([np.matrix([1.5,1]), np.matrix([2.5,1]), np.matrix([3.5,1]), np.matrix([2.5,2]), np.matrix([3.5,2]), np.matrix([4.5,2])])

mu_0 = 1/len(P0)*np.matrix([sum(P0[:,0,0]), sum(P0[:,0,1])])
mu_1 = 1/len(P1)*np.matrix([sum(P1[:,0,0]), sum(P1[:,0,1])])

print('mu_0 = \n', mu_0)
print('mu_1 = \n', mu_1)

S_0 = 0
for x in range(len(P0)):
    S_0 += (P0[x,:]-mu_0).T*(P0[x,:]-mu_0)

print('S_0 = \n', S_0)

S_1 = 0
for x in range(len(P1)):
    S_1 += (P1[x,:]-mu_1).T*(P1[x,:]-mu_1)

print('S_1 = \n', S_1)

S_w = S_0+S_1

print('S_w = \n', S_w)

print('S_w^(-1) = \n', S_w.I)

S_b = (mu_0-mu_1).T * (mu_0-mu_1)

print('S_B = \n', S_b)

lam = S_w.I * (mu_0-mu_1).T
print('lambda = \n', lam)

norm = np.linalg.norm(lam)
print("Normierung: ", norm)
L_norm = lam/norm
print('L_norm = \n', L_norm)


# c)
def f(x):
    return lam[0,0]/lam[1,0]*x+4

xx = np.linspace(0.9, 3.6,100)

plt.plot(P0[:,0,0], P0[:,0,1], 'bx', label=r'$P_0$')
plt.plot(P1[:,0,0], P1[:,0,1], 'rx', label=r'$P_1$')
plt.plot(xx, f(xx), 'k-', label=r'Projektionsgerade $\lambda \vec{e}_\lambda$')

plt.legend()
plt.tight_layout()
plt.savefig('c.pdf')

plt.clf()

# d)
Projektion0 = np.array([])
Projektion1 = np.array([])

for x0, x1 in zip(P0, P1):
    Projektion0 = np.append(Projektion0, np.dot(L_norm.T,x0.T))
    Projektion1 = np.append(Projektion1, np.dot(L_norm.T,x1.T))

print('Projektion0 = \n', Projektion0)
print('Projektion1 = \n', Projektion1)

plt.hist(Projektion0, label='P0 = Signal', bins=30, color='b', alpha=0.8, density=True)
plt.hist(Projektion1, label='P1 = Untergrund', bins=30, color='r', alpha=0.8, density=True)
plt.xlabel('Projektionsebene')
plt.ylabel('Anzahl Ereignisse')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('d.pdf')
plt.clf()

# e)
lcut = -0.362

tp = 5
fp = 1
fn = 1
tn = 5

rein = tp/(tp+fp)
eff = tp/(tp+fn)
genau = (tp+tn)/(tp+fn+tn+fp)

print('Reinheit = %0.2f' % rein)
print('Effizienz = %0.2f' % eff)
print('Genauigkeit = %0.2f' % genau)
