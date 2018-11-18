import numpy as np
import matplotlib.pyplot as plt
np.random.seed = 42

def shift(list, n):
    n = n % len(list)
    return list[n:] + list[:n]

#Funktion zum berechnen des Mittelwerts
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

#a)
covP0 = [[3.5**2, 0.9*3.5*2.6],[0.9*3.5*2.6, 2.6**2]]
P0 = np.random.multivariate_normal([0,3],covP0, size=10000)
x0, y0 = zip(*P0)
plt.plot(x0, y0, 'b.', label='P_0', alpha=0.07)
covP1 = [[3.5**2, 0.75*3.5*1.51],[0.75*3.5*1.51, 1.51**2]]
P1 = np.random.multivariate_normal([6,1.44],covP1, size=10000)
x1, y1 = zip(*P1)
plt.plot(x1, y1, 'r.', label='P_1', alpha=0.1)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Populationen.pdf")

#b)
print("Population 0")
m0=np.array([np.mean(x0), np.mean(y0)])
a, b=zip(*P0)
print("Mittelwerte von P0: ", m0)
print("Varianz von x0: ", np.var(x0))
print("Varianz von y0: ", np.var(y0))
print("Kovarianz cov(x, y): ", np.cov(a,b, ddof=1)[0][1])
print("Korrelation rho: ", np.cov(a,b, ddof=1)[0][1]/np.sqrt(np.var(x0)*np.var(y0)), "\n \n")

print("Population 1")
m1=np.array([np.mean(x1), np.mean(y1)])
a, b=zip(*P1)
print("Mittelwerte von P0: ", m1)
print("Varianz von x0: ", np.var(x1))
print("Varianz von y0: ", np.var(y1))
print("Kovarianz cov(x, y): ", np.cov(a,b, ddof=1)[0][1])
print("Korrelation rho: ", np.cov(a,b, ddof=1)[0][1]/np.sqrt(np.var(x1)*np.var(y1)), "\n \n")

print("Population 0 + Population 1")
P = np.vstack((P0,P1))
x, y = zip(*P)
m2=np.array([np.mean(x), np.mean(y)])
print("Mittelwerte von P0: ", m2)
print("Varianz von x0: ", np.var(x))
print("Varianz von y0: ", np.var(y))
print("Kovarianz cov(x, y): ", np.cov(x,y, ddof=1)[0][1])
print("Korrelation rho: ", np.cov(x,y, ddof=1)[0][1]/np.sqrt(np.var(x)*np.var(y)), "\n \n")
