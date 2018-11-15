import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats
from scipy import linalg

P0_p = pd.read_hdf('zwei_populationen.h5', key='P_0_10000')
P1_p = pd.read_hdf('zwei_populationen.h5', key='P_1')


# Aufgabenteil a) #############################################################
# Mittelwerte bilden
mue0=np.array([P0_p.x.mean(), P0_p.y.mean() ])
mue1=np.array([P1_p.x.mean(), P1_p.y.mean() ])

print('Mittelwerte:')
print('mue0 = ', mue0)
print('mue1 0 ', mue1)
print()

# Aufgabenteil b) #############################################################
# Kovarianzmatrizen berechnen
V_P0 = P0_p.cov()
V_P1 = P1_p.cov()
V_P01 = P0_p.append(P1_p).cov()
print('Kovarianzmatrizen:')
print('V_P0 = ')
print(V_P0)
print('V_P1 = ')
print(V_P1)
print('V_P01 = ')
print(V_P01)
print()

# Aufgabenteil c) #############################################################
# Lineare Fisher-Diskriminante L

# Definition des Vektorprodukts
def vprod(x):
    y = np.atleast_2d(x)
    return np.dot(y.T, y)

S0 = np.zeros(2)
for index, row in (P0_p - mue0).iterrows():
    S0 = S0 + vprod(row)

S1 = np.zeros(2)
for index, row in (P1_p - mue1).iterrows():
    S1 = S1 + vprod(row)

# Addition der Matritzen
SW = S0 + S1
print('S0 = ', S0)
print('S1 = ', S1)
print('SW = ', SW)
print()

L = np.dot(np.linalg.inv(SW), (mue0-mue1))
print('L = ', L)
# Geradengleichung noch ausrechnen!!!
norm = np.linalg.norm(L)
print("Normierung: ", norm)
L_norm = L/norm
print('L_norm = ', L_norm)

# Aufgabenteil d) #############################################################
Projektion_0 = np.array([])
for index, row in (P0_p).iterrows():
    Projektion_0 = np.append(Projektion_0, np.vdot(row, L_norm))

Projektion_1 = np.array([])
for index, row in (P1_p).iterrows():
    Projektion_1 = np.append(Projektion_1, np.vdot(row, L_norm))

plt.hist(Projektion_0, label='P0 = Signal', bins=30, color='b', alpha=0.8)
plt.hist(Projektion_1, label='P1 = Untergrund', bins=30, color='r', alpha=0.8)
plt.xlabel('Projektionsebene')
plt.ylabel('Anzahl Ereignisse')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('Projektionen.pdf')
plt.clf()

# Aufgabenteil e), f) und g) #############################################################
lcut = np.linspace(-5,5, 100)
Back = Projektion_1
Sig = Projektion_0

#leere Arrays erzeugen
eff = np.zeros(len(lcut))
rein = np.zeros(len(lcut))
ver = np.zeros(len(lcut))
sign = np.zeros(len(lcut))
#Effizienz und Reinheit für jedes l_cut berechnen
for i in range(len(lcut)):
    tp = len(Sig[Sig > lcut[i]])
    fp = len(Back[Back > lcut[i]])
    fn = len(Sig[Sig <= lcut[i]])

    eff[i] = tp/(tp + fn)
    rein[i] = tp/(tp + fp)
    if fp != 0:
            ver[i] = tp/fp
    sign[i] = tp/(np.sqrt(tp + fp))

# Maximum des Verhältnisses berechnen
lcut_maxv = lcut[np.argmax(ver)]
print('lcut_maxv = ', lcut_maxv )

# Maximum der Signifikanz berechnen
lcut_maxs = lcut[np.argmax(sign)]
print('lcut_maxs = ', lcut_maxs)


# Plots
plt.plot(lcut, eff, 'r-', label='Effizienz')
plt.plot(lcut, rein, 'b-', label='Reinheit')
plt.xlabel(r"$\lambda_{cut}$")
plt.ylabel('Effizienz und Reinheit')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('EffizienzReinheit.pdf')
plt.clf()

plt.plot(lcut, ver, 'y-', label='Verhältnis')
plt.xlabel(r"$\lambda_{cut}$")
plt.ylabel('Signal/Backround')
plt.axvline(x=lcut_maxv, linestyle='--', label='Maximum')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('Verhältnis.pdf')
plt.clf()

plt.plot(lcut, sign, 'r-', label='Signifikanz')
plt.axvline(x=lcut_maxs, linestyle='--', label='Maximum')
plt.xlabel(r"$\lambda_{cut}$")
plt.ylabel('Signifikanz')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('Signifikanz.pdf')
plt.clf()


# Aufgabenteil h) #############################################################
P2_p = pd.read_hdf('zwei_populationen.h5', key='P_0_1000')

mue2=np.array([P2_p.x.mean(), P2_p.y.mean() ])
print('Mittelwerte:')
print('mue2 = ', mue2)

V_P2 = P2_p.cov()
V_P21 = P2_p.append(P1_p).cov()

print('Kovarianzmatrizen:')
print('V_P2 = ')
print(V_P2)
print('V_P21 = ')
print(V_P21)
print()

S2 = np.zeros(2)
for index, row in (P2_p - mue2).iterrows():
    S2 = S2 + vprod(row)

# Addition der Matritzen
SW2 = S2 + S1
print('S2 = ', S2)
print('SW2 = ', SW2)
print()

L2 = np.dot(np.linalg.inv(SW2), (mue2-mue1))
print('L2 = ', L2)
# Geradengleichung noch ausrechnen!!!
norm2 = np.linalg.norm(L2)
print("Normierung2: ", norm2)
L2_norm = L2/norm2
print('L2_norm = ', L2_norm)

Projektion2_2 = np.array([])
for index, row in (P2_p).iterrows():
    Projektion2_2 = np.append(Projektion2_2, np.vdot(row, L2_norm))

Projektion2_1 = np.array([])
for index, row in (P1_p).iterrows():
    Projektion2_1 = np.append(Projektion2_1, np.vdot(row, L2_norm))

plt.hist(Projektion2_2, label='P2 = Signal', bins=30, color='b', alpha=0.8)
plt.hist(Projektion2_1, label='P1 = Untergrund', bins=30, color='r', alpha=0.8)
plt.xlabel('Projektionsebene')
plt.ylabel('Anzahl Ereignisse')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('Projektionen2.pdf')
plt.clf()


Back2 = Projektion2_1
Sig2 = Projektion2_2

#leere Arrays erzeugen
eff2 = np.zeros(len(lcut))
rein2 = np.zeros(len(lcut))
ver2 = np.zeros(len(lcut))
sign2 = np.zeros(len(lcut))
#Effizienz und Reinheit für jedes l_cut berechnen
for i in range(len(lcut)):
    tp2 = len(Sig2[Sig2 > lcut[i]])
    fp2 = len(Back2[Back2 > lcut[i]])
    fn2 = len(Sig2[Sig2 <= lcut[i]])

    eff2[i] = tp2/(tp2 + fn2)
    rein2[i] = tp2/(tp2 + fp2)
    if fp2 != 0:
            ver2[i] = tp2/fp2
    sign2[i] = tp2/(np.sqrt(tp2 + fp2))

# Maximum des Verhältnisses berechnen
lcut2_maxv = lcut[np.argmax(ver2)]
print('lcut2_maxv = ', lcut2_maxv )

# Maximum der Signifikanz berechnen
lcut2_maxs = lcut[np.argmax(sign2)]
print('lcut2_maxs = ', lcut2_maxs)


# Plots
plt.plot(lcut, eff2, 'r-', label='Effizienz')
plt.plot(lcut, rein2, 'b-', label='Reinheit')
plt.xlabel(r"$\lambda_{cut}$")
plt.ylabel('Effizienz und Reinheit')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('EffizienzReinheit2.pdf')
plt.clf()

plt.plot(lcut, ver2, 'y-', label='Verhältnis')
plt.xlabel(r"$\lambda_{cut}$")
plt.ylabel('Signal/Backround')
plt.axvline(x=lcut2_maxv, linestyle='--', label='Maximum')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('Verhältnis2.pdf')
plt.clf()

plt.plot(lcut, sign2, 'r-', label='Signifikanz')
plt.axvline(x=lcut2_maxs, linestyle='--', label='Maximum')
plt.xlabel(r"$\lambda_{cut}$")
plt.ylabel('Signifikanz')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('Signifikanz2.pdf')
plt.clf()
