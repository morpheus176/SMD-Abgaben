import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(42)


#Erzeugt Zufallszahlen nach der Definition der Aufgabe. fac gibt an, wie oft durchgegangen werden soll (fac*m mal)
def random (a, b, m, x_0, fac):
    x_a=x_0
    List=[None]
    for i in range(fac*m):
            x_n=((a*x_a+b) % m)
            List[i]=x_n/m
            List+=[None]
            x_a=x_n
    List=List[:-1]
    return(List)

#Funktion zum finden von Periodenlängen
def findperiod(list):
    #für ein x...
    for x in range(len(list)):
        #---suche ein xx mit List[x]==List[xx+x]
        for xx in range(len(list)-x):
            if list[x] == list[xx+x] and xx != 0:
                #wenn gefunden, dann überprüfe, ob List[x+i]==List[xx+x+i]
                for i in range(len(list)-x-xx):
                    #Tritt ungleichheit auf => brich ab
                    if list[x+i] != list [xx+x+i]:
                        break
                    #keine Ungleichheiten bis zum Ende der Liste: Periode gefunden!
                    elif i==(len(list)-xx-x-1):
                        return xx

#Funktion zum shiften von Arrayeinträgen
def shift(list, n):
    n = n % len(list)
    return list[n:] + list[:n]

#Teilaufagbe a)
#stelle Periodenlängen von  0 bis 42 in einem Plot da.
x0=0
for a in range(43):
    plt.plot(a, findperiod(random (a, 3, 1024, x0, 2)), 'r.')
plt.savefig("Teilaufgabe_a).pdf")
plt.clf()
plt.cla()
#Teilaufgabe b)
#Erstelle histogramm für m=10000, b=3456 und a=1601
Data=random(1601, 3456, 10000, 0, 1)
plt.hist(Data, density=True, color='grey', bins=25)
plt.xlabel("Zufallszahl")
plt.ylabel("Anzahl")
plt.title("25 Bins")
plt.savefig("Teilaufgabe_b)_Histogramm.pdf")
plt.clf()

#Teilaufgabe c)
#2D-Scatter
x=random(1601, 3456, 10000, 0, 1)
y = shift(x, 1)
z=shift(Data, 2)
plt.scatter(x, y)
plt.xlabel(r'$x_{i}$')
plt.ylabel(r'$x_{i+1}$')
plt.savefig("Teilaufgabe_c)_2D-Scatter.pdf")
plt.clf()

#3D-Scatter
fig = plt.figure()
ax=Axes3D(fig)

ax.scatter(x, y, z, alpha=0.3)
ax.set_xlabel(r'$x_{i}$')
ax.set_ylabel(r'$x_{i+1}$')
ax.set_zlabel(r'$x_{i+2}$')
plt.show()
#plt.savefig("Test.pdf")
plt.clf()

#Teilaufgabe d)
newdata=[]
#Erstelle 10000 Zufallszahlen
for x in range(10000):
    newdata.append(np.random.uniform(0, 1))

#Erstelle Histogramm
plt.hist(newdata, density=True, color='grey', bins=25)
plt.xlabel("Zufallszahl")
plt.ylabel("Anzahl")
plt.title("25 Bins")
plt.savefig("Teilaufgabe_d)_Histogramm.pdf")
plt.clf()

#Erstelle Scatter
x=newdata
y=shift(newdata, 1)
z=shift(newdata, 2)

#Erstelle 2D Scatter
plt.scatter(x, y, alpha=0.3)
plt.xlabel(r'$x_{i}$')
plt.ylabel(r'$x_{i+1}$')
plt.savefig("Teilaufgabe_d)_2D-Scatter.pdf" )
plt.clf()
plt.cla()

#3D Scatter
#ich weiß nicht wieso, aber ich muss das anscheinend neu definieren, sonst bleiben meine Plots leer .
fig = plt.figure()
ax=Axes3D(fig)
#x=newdata
#y=shift(newdata, 1)
#z=shift(newdata, 2)
ax.scatter(x, y, z, alpha=0.3)
ax.set_xlabel(r'$x_{i}$')
ax.set_ylabel(r'$x_{i+1}$')
ax.set_zlabel(r'$x_{i+2}$')
plt.show()
plt.clf()
#print(x)

#Teilaufgabe e)
#Überprüfe für 1^4 Startwerte, wie oft 1/2 vorkommt und schreibe Ergebnisse in txt
datacount=open("datalog.txt", "w")
counter=[]
for x in range(10000):
    counter.append(random(1, 3, 1024, x/1000,  1).count(0.5))
    string="Fuer x_0 = " + str(x/1000) +" kommt der Wert 0.5 " + str(counter[x]) + " mal vor \n"
    datacount.write(string)
