import numpy as np
import matplotlib.pyplot as plt

def quer_V1(x):
    return a**2/s*((2+np.sin(x)**2)/(1-b**2*np.cos(x)**2))

def quer_V2(x):
    return a**2/s*((2+np.sin(x)**2)/gamma**(-2)*np.cos(x)**2+np.sin(x)**2)
def konditionsquer(x):
    return abs(((3*b**2-1)*x*np.sin(2*x))/((np.sin(x)**2+2)*(b**2*np.cos(x)**2-1)))

E=50*10**9
m=511*10**6
a=1/137
gamma=E/m
s=(2*E)**2
b=np.sqrt(1-gamma**(-2))

x=np.linspace(0, np.pi, 5000)
plt.plot(x, quer_V1(x)*10**(22), 'r-', label='ohne Umformung')
plt.plot(x, quer_V2(x)*10**(22), 'b-', label='mit Umformung')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\frac{d \sigma}{d \Omega}/[10^{-22} m^2]$')
plt.xticks([0, np.pi / 2, np.pi],
           [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$"])
plt.legend()
plt.savefig("diffquerschnitt.pdf")

plt.clf()
plt.plot(x, konditionsquer(x))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\kappa$')
plt.xticks([0, np.pi / 2, np.pi],
           [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$"])
plt.savefig('Konditionierungszahl.pdf')
