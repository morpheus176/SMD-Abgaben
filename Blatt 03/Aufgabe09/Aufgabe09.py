import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats


x_i=30 # Startwert
s=2 # Schrittweite
Anzahl_Zufallsvariablen= 10**5
Zufallsvariablen = np.array([])
Zufallsvariablen = np.append(Zufallsvariablen, x_i) # Startwert in das Array speichern


# Funktion einlesen
N= 15/(np.pi**4) # Normierungsfaktor
def f(x):
    return N*x**3/(np.exp(x)-1)


np.random.seed(100)


for i in range(Anzahl_Zufallsvariablen-1):
    # Nächsten Schritt ziehen
    if x_i<s:
        x_j= np.random.uniform(0, x_i+s, 1)
    else:
        x_j= np.random.uniform(x_i-s, x_i+s, 1)

    #Übergangswahrscheinlichkeit p
    p= np.min([1, f(x_j) / f(x_i) ])

    e = np.random.uniform(0,1,1)
    if e>p:
        x_i = x_i
    else:
        x_i = x_j

    # Zufallsvariable dem Array hinzufügen
    Zufallsvariablen = np.append(Zufallsvariablen, x_i)



# Plots
L1plot = np.linspace(np.min(Zufallsvariablen), np.max(Zufallsvariablen), 1000)
k = np.argmax(f(L1plot))

plt.hist(Zufallsvariablen, bins=np.linspace(np.min(Zufallsvariablen), np.max(Zufallsvariablen),100), density=True, label='normiertes Histogramm')
plt.plot(L1plot, f(L1plot) , 'r-', label='Planck-Verteilung')
plt.legend(loc="best")
plt.xlabel(r"$Zufallszahl$")
plt.ylabel(r"$relative Wahrscheinlichkeit$")
plt.tight_layout()
plt.savefig('Metropolis.pdf')
plt.clf()

i = np.arange(1,Anzahl_Zufallsvariablen+1)
plt.plot(i, Zufallsvariablen, '.', markersize=1, alpha=0.5)
plt.axhline(y=L1plot[k], color='r', label='Maximum der Verteilungsfunktion')
plt.xlabel(r"$Schritt$")
plt.ylabel(r"$Zufallszahl$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('MetropolisSchritte.pdf')
plt.clf()

plt.plot(i[0:200], Zufallsvariablen[0:200], '-', alpha=0.8)
plt.axhline(y=L1plot[k], color='r')
plt.xlabel(r"$Schritt$")
plt.ylabel(r"$Zufallszahl$")
plt.tight_layout()
plt.savefig('MetropolisSchritte2.pdf')
plt.clf()
