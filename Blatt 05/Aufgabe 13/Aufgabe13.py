import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

DIRECTORY = "build/"
SAMPLE_SIZE = 100000

if not os.path.exists(DIRECTORY):
	os.makedirs(DIRECTORY)

## Dataframe:
df = pd.DataFrame()

### Aufgabenteil a)
GAMMA = 2.7
EMIN = 1
norm = (GAMMA-1)*(EMIN)**(GAMMA-1)

# Invertiertes Integral
def Ginv(x):
	return ((1-GAMMA)/norm*x + (EMIN)**(1-GAMMA))**(1/(1-GAMMA))

# Generiere size Zahlen
def genE(size):
    r = np.random.uniform(0, 1, size)
    E = Ginv(r)
    return E

# Energiewerte
E = genE(SAMPLE_SIZE)

# Key Energy
df['Energy'] = E
print(df)

### Aufgabenteil b)

# Detektionswahscheinlichkeit
def P(E):
    return (1-np.exp(-E/2))**3

PMAX = 1

def genMask(size, E):
    Mask = np.array([])
    for i in range(0, size):
        u = np.random.uniform(0, PMAX, 1)
        if P(E[i]) <= u:
            Mask = np.append(Mask, False)
        else:
            Mask = np.append(Mask, True)
    return Mask

Mask = genMask(SAMPLE_SIZE, E)

# Key Energy
df['AcceptanceMask'] = Mask
print(df)

#normalverteilung
def normal(size, mu, sig):
	x1 = np.array([])
	x2 = np.array([])
	while x1.size < size:
		u1 = np.random.uniform(0, 1)
		u2 = np.random.uniform(0, 1)
		v1 = 2*u1 - 1
		v2 = 2*u2 - 1
		s = v1**2 + v2**2
		if s < 1:
			x1 = np.append(x1, v1*np.sqrt(-2/s*np.log(s)))
			x2 = np.append(x2, v2*np.sqrt(-2/s*np.log(s)))
	return np.sqrt(sig)*x1+mu, np.sqrt(sig)*x2+mu

plt.hist(normal(20000,5,5), bins=20)
plt.show()
