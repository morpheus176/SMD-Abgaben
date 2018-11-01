import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats

#Aufgabe 5 a)
np.random.seed(12)
uniform1 = np.random.uniform(0, 1, 1000000)
x_min1 = -100
x_max1 = 100
uniform1_transformiert = uniform1*(x_max1- x_min1) +x_min1
plt.ylim(0, 0.006)
plt.hist(uniform1_transformiert, bins=20, density=True)
plt.xlabel('x(U)')
plt.ylabel('relative Häufigkeit x(U)')
plt.savefig('Transformierte1.pdf')
plt.clf()

#b)
uniform2 = np.random.uniform(0, 1, 10000)
tau = 10
uniform2_transformiert = -tau * np.log(1-uniform2)
plt.hist(uniform2_transformiert, bins=np.linspace(0,60,20), density=True)
plt.xlabel('t(U)')
plt.ylabel('relative Häufigkeit t(U)')
plt.savefig('Transformierte2.pdf')
plt.clf()

#c)
uniform3 = np.random.uniform(0, 1, 100000)
xmin3= 1
xmax3= 10
n=2
uniform3_transformiert = ( uniform3*(xmax3**(1-n)-xmin3**(1-n))+xmin3**(1-n) )**(1/(1-n))
plt.xlim(xmin3, xmax3)
plt.hist(uniform3_transformiert, bins=20, density=True)
plt.xlabel('x(U)')
plt.ylabel('relative Häufigkeit x(U)')
plt.savefig('Transformierte3.pdf')
plt.clf()

#d)
uniform4 = np.random.uniform(0,1, 1000000)
uniform4_transformiert = np.tan(np.pi*(uniform4-1/2))
plt.xlim(-10,10)
plt.hist(uniform4_transformiert, bins=np.linspace (-10 , 10, 30), density=True)
plt.xlabel('x(U)')
plt.ylabel('relative Häufigkeit x(U)')
plt.savefig('Transformierte4.pdf')
plt.clf()

#e)
data = np.load("empirisches_histogramm.npy")
plt.hist(data['bin_mid'], bins=np.linspace(0, 1, 51) ,weights =data['hist'])
plt.show ()
