import numpy as np
import matplotlib.pyplot as plt
import funktionen as funkt

x=np.linspace(0, 80*10**3, 500000)
y=funkt.Fehler(funkt.f, x)*100
z=funkt.Fehler(funkt.g, x)*100

plt.plot(np.log(x), np.log(y))
#plt.xscale("log")
plt.xlabel(r'$\log{x}$')
plt.ylabel('log(Abweichung in %)')
plt.savefig("Fehler1a.pdf")

plt.clf()
plt.plot(x, z*10**12)
plt.xlabel(r'$x$')
plt.ylabel('Abweichung in %'r'$ \cdot 10^{12}$')
plt.xscale("log")
plt.savefig("Fehler1b.pdf")
