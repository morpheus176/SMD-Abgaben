from sympy.solvers import solve
from sympy import Symbol
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import timeit

np.random.seed(42)

N=15/(np.pi)**4

def f(x):
    try:
        return N*x**3/(np.e**x-1)
    except ZeroDivisionError:
        return 0

max_x = optimize.fmin(lambda x: -f(x), 0, disp=False)

#gibt den maximalwert der Funktion f(x) aus. Als Funktion damit Rejection damit arbeiten kann.
def fmax(xx):
    return f(max_x)

def rejection(funktion, Start, Ende):
    xin=[]
    yin=[]
    xout=[]
    yout=[]

    while len(xin)<10**5:
        xx=np.random.uniform(Start, Ende)
        yy=np.random.uniform(Start, funktion(xx))
        if yy <= f(xx):
            xin.append(xx)
            yin.append(yy)
        else:
            xout.append(xx)
            yout.append(yy)

    return xin, yin, xout, yout

def g(x):
    return 200*N*x**(-0.1)*np.exp(-x**0.9)

def h(x):
    return f(max_x) - g(x)

x_s = brentq(h, 5, 10)

def gg(x):
    if x < x_s:
        return f(max_x)
    else:
        return 200*N*x**(-0.1)*np.exp(-x**0.9)

# a)

start_time = timeit.default_timer()

xin, yin, xout, yout = rejection(fmax, 0, 20)

laufzeit_a = timeit.default_timer() - start_time

x=np.linspace(0, 20, 500)
plt.plot(xin, yin, 'b.', markersize=0.05)# alpha=0.01)
plt.plot(xout, yout, 'r.', markersize=0.05)# alpha=0.01)
plt.plot(x, f(x), 'k-', label=r'Planck-Verteilung')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("a.pdf")
plt.clf()

verworfen_a = len(xout)


# b)
start_time = timeit.default_timer()

xin, yin, xout, yout = rejection(gg, 0, 20)

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
print('\n Verhaeltnisse a)/b): \n verworfene Ereignisse: %0.2f \n Laufzeit: %0.2f' % (verworfen_a/verworfen_b, laufzeit_a/laufzeit_b))
