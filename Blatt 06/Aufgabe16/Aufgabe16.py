import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats

Temperatur, Wetter, Luftfeuchtigkeit, Wind, Fußball = np.genfromtxt('Tabelle.txt', unpack=True)


def get_Entropie(p_1, p_2):
    if p_1==0:
        S = -p_2*np.log2(p_2)
    elif p_2==0:
        S = -p_1*np.log2(p_1)
    else:
        S = -(p_1*np.log2(p_1) + p_2*np.log2(p_2))
    return S


# Entropie berechnen:
p_Fußball_False = len(Fußball[Fußball==False])/len(Fußball)
p_Fußball_True = 1- p_Fußball_False
S = get_Entropie(p_Fußball_False, p_Fußball_True)


def get_IG(D, Schnitt):
    IG = np.zeros(len(Schnitt))


    for l in range(len(Schnitt)):
        D_u =  len(D[D < Schnitt[l]]) # Anzahl der Werte unter dem Schnitt
        D_ü = len(D)-D_u # Anzahl der Werte über dem Schnitt


        if D_u==0 or D_u==len(D):
            IG[l]=0 # Falls alle Werte unter oder über dem Schnitt sind, ist
                    # der Informationsgewinn 0
        else:
            # Berechnung der bedingten Wahrscheinlichkeiten p_unterSchnitt:
            Anzahl_unterSchnitt_False = 0


            for i in range(len(D)):
                if D[i]<Schnitt[l] and Fußball[i]==False:
                    Anzahl_unterSchnitt_False = Anzahl_unterSchnitt_False+1


            p_unterSchnitt_False = Anzahl_unterSchnitt_False/D_u
            p_unterSchnitt_True = 1-p_unterSchnitt_False
            S_1 = get_Entropie(p_unterSchnitt_False, p_unterSchnitt_True)

            # Berechnung der Wahrscheinlichkeit p_überSchnitt:
            Anzahl_überSchnitt_False = 0


            for i in range(len(D)):
                if D[i]>=Schnitt[l] and Fußball[i]==False:
                    Anzahl_überSchnitt_False = Anzahl_überSchnitt_False+1


            p_überSchnitt_False=Anzahl_überSchnitt_False/D_ü
            p_überSchnitt_True = 1-p_überSchnitt_False
            S_2 = get_Entropie(p_überSchnitt_False, p_überSchnitt_True)

            IG[l] = S - D_u/len(D)*S_1 - D_ü/len(D)*S_2

    return IG

# IG Wetter
#Schnitt_Wetter = np.array([0,1,2,3])
Schnitt_Wetter = np.linspace(0, 3, 100)
IG_Wetter = get_IG(Wetter, Schnitt_Wetter)
print('WETTER:')
print('Bester Schnitt: ', Schnitt_Wetter[np.argmax(IG_Wetter)])
print('Bester IG: ', np.max(IG_Wetter))

# IG Temperatur:
Schnitt_Temperatur = np.linspace(16, 30, 100)
IG_Temperatur = get_IG(Temperatur, Schnitt_Temperatur)
print('TEMPERATUR:')
print('Bester Schnitt: ', Schnitt_Temperatur[np.argmax(IG_Temperatur)])
print('Bester IG: ', np.max(IG_Temperatur))

# IG Luftfeuchtigkeit
Schnitt_Luft = np.linspace(60, 100, 100)
IG_Luft = get_IG(Luftfeuchtigkeit, Schnitt_Luft)
print('LUFTFEUCHTIGKEIT:')
print('Bester Schnitt: ', Schnitt_Luft[np.argmax(IG_Luft)])
print('Bester IG: ', np.max(IG_Luft))

# Plot: Wetter
plt.plot(Schnitt_Wetter, IG_Wetter)
plt.xlabel('Schnitt')
plt.ylabel('IG')
plt.tight_layout()
plt.savefig('IG_Wetter.pdf')
plt.clf()

# Plot: Temperatur
plt.plot(Schnitt_Temperatur, IG_Temperatur)
plt.xlabel('Schnitt')
plt.ylabel('IG')
plt.tight_layout()
plt.savefig('IG_Temperatur.pdf')
plt.clf()

# Plot: Luftfeuchtigkeit
plt.plot(Schnitt_Luft, IG_Luft)
plt.xlabel('Schnitt')
plt.ylabel('IG')
plt.tight_layout()
plt.savefig('IG_Luft.pdf')
plt.clf()
