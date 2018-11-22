import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a)
gamma = 2.7
phi_0 = 1.7

def phi(x):
    return phi_0 * x**(-gamma)

def Phi(x):
    return phi_0/(gamma-1) * (1-x**(-(gamma-1)))

def Phi_inv(x):
    return 1/(1-x)**(1/(gamma-1))

np.random.seed(10)
r = np.random.uniform(0,1, 100000)

energy = Phi_inv(r)
dfEnergy = pd.DataFrame({'Energy': energy})


# b)
def P(x):
    return (1-np.exp(-x/2))**3

np.random.seed(42)
xx = np.random.uniform(0,1, 100000)

acceptance = xx < P(energy)
detected = energy[acceptance == True]

dfAcceptance = pd.DataFrame({'AcceptanceMask': acceptance})

x = np.linspace(1, 1000, 10000)

plt.hist(energy, density=True, bins=np.logspace(0,3), histtype='step', label='gezogene Energien')
plt.hist(detected, density=True, bins=np.logspace(0,3), histtype='step', label='detektierte Energien')
plt.plot(x, phi(x), label=r'$\varphi(E)$')

plt.xlabel(r'$E \:/\: \mathrm{TeV}$')

plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.tight_layout()
plt.savefig('b.pdf')

plt.clf()

# c)

np.random.seed(42)
def polar(size, Energy):
    E=[]
    while len(E)<size:
        u1=np.random.uniform(0,1)
        u2=np.random.uniform(0,1)
        v1=2*u1-1
        v2=2*u2-1
        s=v1**2+v2**2
        if s <= 1:
            x1 = v1*np.sqrt(-2/s*np.log(s))
            x2 = v2*np.sqrt(-2/s*np.log(s))
            p1 = int(np.sqrt(2*Energy[len(E)-1])*x1+10*Energy[len(E)-1])
            p2 = int(np.sqrt(2*Energy[len(E)-1])*x2+10*Energy[len(E)-1])

        if p1>0:
            E.append(p1)

    return E

Hits=polar(10**5, energy)

dfHits = pd.DataFrame({'NumberOfHits' :Hits})

# d)

np.random.seed(25)
def detector(size, Hits):
    x=[]
    y=[]
    while len(x)<size:
        u1=np.random.uniform(0,1)
        u2=np.random.uniform(0,1)
        v1=2*u1-1
        v2=2*u2-1
        s=v1**2+v2**2
        if s <= 1:
            xx=v1*np.sqrt(-2/s*np.log(s))/(np.log10(Hits[len(x)-1]+1))+7
            yy=v2*np.sqrt(-2/s*np.log(s))/(np.log10(Hits[len(x)-1]+1))+3
            if 0 <= xx <= 10 and 0 <= yy <= 10:
                x.append(xx)
                y.append(yy)

    return x, y

xx, yy=detector(10**5, Hits)
plt.grid()
plt.hist2d(xx,yy, bins=[100,100], range=[[0,10],[0,10]], cmap='inferno')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('d.pdf')

plt.clf()

# e)

np.random.seed(44)
size = int(1e7)

mu = 2
sig = 1

log10hits = np.random.normal(mu, sig, size)
hits = 10**(log10hits)
dfBackHits = pd.DataFrame({'NumberOfHits': hits})

plt.hist(log10hits, bins=50, density=True, label=r'Untergrund')
plt.xlabel(r'$\log_{10}z{(\mathrm{Hits})}$')
plt.legend()
plt.tight_layout()
plt.savefig('e_hits.pdf')

plt.clf()

np.random.seed(4)

def ort(mu, sig, rho, size):
    x=[]
    y=[]
    while len(x)<size:
        xx = np.random.normal(0, 1)
        yy = np.random.normal(0, 1)

        xx = np.sqrt(1-rho**2)*sig*xx+rho*sig*yy+mu
        yy = sig*yy+mu

        if 0 <= xx <= 10 and 0 <= yy <= 10:
            x.append(xx)
            y.append(yy)

    return x, y

mu = 5
sig = 3
rho = 0.5

x, y = ort(mu, sig, rho, size)

plt.grid()
plt.hist2d(x,y, bins=[100,100], range=[[0,10],[0,10]], cmap='inferno')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.savefig('e_detektor.pdf')
plt.clf()

dfBackX = pd.DataFrame({'x': x})
dfBackY = pd.DataFrame({'y': y})

dfBackground = pd.concat([dfBackHits, dfBackX, dfBackY], axis=1)
dfBackground = dfBackground.to_hdf('NeutrinoMC.hdf5', key='Background')
