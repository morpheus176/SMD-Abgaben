import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


np.random.seed(100)

def softmax(f):
    fexp = np.exp(f)
    return fexp/np.sum(fexp, axis=1, keepdims=True)


# Einlesen
P0 = pd.read_hdf('populationen.hdf5', key='P_0')
P1 = pd.read_hdf('populationen.hdf5', key='P_1')

# Zusammenfassen und Labels
Px = np.append(P0.x, P1.x)
Py = np.append(P0.y, P1.y)
Label = np.append(np.zeros(len(P0)), np.ones(len(P1)))
Population = np.vstack([Px, Py, Label])
input = Population.T[:, :2]
##Training
rate = 0.5
Epochen = 100

W = np.random.rand(2, 2)
b = np.random.rand(2)

N = len(Px)

for i in range(Epochen):
    f_i = input @ W + b

    #Gradienten berechnen
    df_i = softmax(f_i)
    df_i[range(N), [int(d) for d in Population.T[:, 2]]] -=1
    df_i /= N

    dW = input.T @ df_i
    db = np.array([np.sum(df_i[:, 0]), np.sum(df_i[:, 1])])

    W -= rate * dW
    b -= rate * db

print(W)
print(b)

def Gerade(X, W, b):
    return (b[1]-b[0]+X*(W[0][1]-W[1][1]))/(W[1][0]-W[0][0])

xPlot = np.linspace(np.min(Px), np.max(Px), 10000)
plt.plot(P0.x, P0.y, 'r.', alpha=0.7, markersize=0.7, label='P0')
plt.plot(P1.x, P1.y, 'b.', alpha=0.7, markersize=0.7, label='P1')
plt.plot(xPlot, Gerade(xPlot, W, b), 'black', label='Gerade')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('Plot.pdf')
