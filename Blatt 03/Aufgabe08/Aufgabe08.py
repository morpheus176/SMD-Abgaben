from sympy.solvers import solve
from sympy import Symbol
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import timeit

np.random.seed(42)

N=15/(np.pi)**4


# a)

def f(x):
    try:
        return N*x**3/(np.e**x-1)
    except ZeroDivisionError:
        return 0

max_x = optimize.fmin(lambda x: -f(x), 0)

xin=[]
yin=[]
xout=[]
yout=[]

start_time = timeit.default_timer()

while len(xin)<10**5:
    xx=np.random.uniform(0, 20)
    yy=np.random.uniform(0, f(max_x))
    if yy <= f(xx):
        xin.append(xx)
        yin.append(yy)
    else:
        xout.append(xx)
        yout.append(yy)


laufzeit_a = timeit.default_timer() - start_time

x=np.linspace(0, 20, 500)
plt.plot(xin, yin, 'b.', markersize=0.05)# alpha=0.01)
plt.plot(xout, yout, 'r.', markersize=0.05)# alpha=0.01)
plt.plot(x, f(x), 'k-', label=r'Planck-Verteilung')
#plt.hlines(y=f(max_x), xmin=0, xmax=20, linewidth=1, color='black')
#plt.vlines(x=max(x), ymin=0, ymax=f(max_x), linewidth=1, color='black')
#plt.vlines(x=max(-x), ymin=0, ymax=f(max_x), linewidth=1, color='black')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("a.pdf")

plt.clf()

verworfen_a = len(xout)


# b)

def g(x):
    return 200*N*x**(-0.1)*np.exp(-x**0.9)

def h(x):
    return f(max_x) - g(x)

x_s = brentq(h, 5, 10)

xin=[]
yin=[]
xout=[]
yout=[]

start_time = timeit.default_timer()

while len(xin)<10**5:
    xx=np.random.uniform(0, 15)
    if xx <= x_s:
        yy=np.random.uniform(0, f(max_x))
        if yy <= f(xx):
            xin.append(xx)
            yin.append(yy)
        else:
            xout.append(xx)
            yout.append(yy)
    else:
        yy=np.random.uniform(0, g(xx))
        if yy <= f(xx):
            xin.append(xx)
            yin.append(yy)
        else:
            xout.append(xx)
            yout.append(yy)


laufzeit_b = timeit.default_timer() - start_time

verworfen_b = len(xout)

x = np.linspace(0, 15, 500)
plt.plot(xin, yin, 'b.', markersize=0.05)
plt.plot(xout, yout, 'r.', markersize=0.05)
plt.plot(x, f(x), 'k-', label=r'Planck-Verteilung')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.legend()
plt.tight_layout()
plt.savefig('b.pdf')


# c)

print('\n Anzahl der verworfenen Ereignisse: \n a) %0.0f \n b) %0.0f' % (verworfen_a, verworfen_b))
print('\n Laufzeit: \n a) %0.2f s \n b) %0.2f s' % (laufzeit_a, laufzeit_b))
print('\n Verhältnisse a)/b): \n verworfene Ereignisse: %0.2f \n Laufzeit: %0.2f' % (verworfen_a/verworfen_b, laufzeit_a/laufzeit_b))
