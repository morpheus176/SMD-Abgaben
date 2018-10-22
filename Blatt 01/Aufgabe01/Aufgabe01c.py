import numpy as np
import matplotlib.pyplot as plt
import funktionen as funkt

x=np.linspace(1e-7, 0.001, 500000)
x2=np.linspace(1e-10, 80*10**3, 500000)
y=funkt.Fehler(funkt.f, x2)*100
z=funkt.Fehler(funkt.g2, x)*100

#x=np.linspace(-5.5, 5.5, 500000)
plt.plot(np.log(x), z)
plt.xlabel(r'$\log{(x)}$')
plt.ylabel(r'Abweichung in %')
#plt.xscale("log")
plt.savefig("Fehler1b.pdf")
#plt.show()

plt.clf()
#plt.plot(np.log(x2), np.log(y))
plt.plot(x2*10**(-3), y)
#plt.xscale("log")
plt.xlabel(r'$x \: \cdot \: 10^3$')
plt.ylabel('Abweichung in %')
plt.savefig("Fehler1a.pdf")
#plt.show()
