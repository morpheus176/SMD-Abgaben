from sympy.solvers import solve
from sympy import Symbol
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def f(x):
    try:
        N=15/(np.pi)**4
        return N*x**3/(np.e**x-1)
    except ZeroDivisionError:
        return 0

max_x = optimize.fmin(lambda x: -f(x), 0)

xin=[]
yin=[]
xout=[]
yout=[]
while len(xin)<10**5:
    xx=np.random.uniform(0, 20)
    yy=np.random.uniform(0, f(max_x))
    if yy <= f(xx):
        xin.append(xx)
        yin.append(yy)
    else:
        xout.append(xx)
        yout.append(yy)


x=np.linspace(0, 20, 500)
plt.plot(x, f(x))
plt.plot(xin, yin, 'b.', alpha=0.01)
plt.plot(xout, yout, 'r.', alpha=0.01)
plt.hlines(y=f(max_x), xmin=0, xmax=20, linewidth=1, color='black')
plt.vlines(x=max(x), ymin=0, ymax=f(max_x), linewidth=1, color='black')
plt.vlines(x=max(-x), ymin=0, ymax=f(max_x), linewidth=1, color='black')
plt.savefig("Teilaufgabe_a).pdf")
