import numpy as np
import time

def f(x):
    return (x**3+1/3)-(x**3-1/3)

def g(x):
    try:
        return ((3+x**3/3)-(3-x**3/3))/x**3
    except ZeroDivisionError:
        #print("ZeroDivisionError fuer x= ", x, ".")
        return 2/3
def gg(x):
    return ((3+x**3/3)-(3-x**3/3))/x**3

def check(grenze, funktion, dezimal, abweichungen):
    start = time.clock()
    if dezimal <0:
        print("Die Anzahl der Dezimalstellen muss größer als 0 sein!")
        return 0
    Wert=2/3
    gefunden=0
    if grenze < 0:
        Grenze=-grenze
    else:
        Grenze=grenze
    print("Fuer ein x von 0 bis ", grenze, " mit ", dezimal, " Dezimalstellen ergeben sich folgende Abweichungen: \n")

    for x in range(grenze*10**dezimal):
        if grenze < 0:
            x=-x

        if Fehler(funktion, x*10**(-dezimal))>0.01 and gefunden!=abweichungen : #or abs((2/3-f(x))/(2/3))<0.99:
            if Wert != f(x*10**(-dezimal)):
                print("x=", x*10**(-dezimal), "f(x)=", funktion(x*10**(-dezimal)), ", Fehler : ", Fehler(funktion,x*10**(-dezimal))*100, "%")
                gefunden+=1
            Wert=funktion(x*10**(-dezimal))

        elif funktion(x*10**(-dezimal))==0 and gefunden == abweichungen:
            print("x=", x*10**(-dezimal), "f(x)=", funktion(x*10**(-dezimal)), ", Fehler : ", Fehler(funktion, x*10**(-dezimal))*100, "%")
            break
    print("Dauer= ", time.clock()-start, "s \n")

def doublecheck(grenze, funktion, dezimal):
    check(grenze, funktion, dezimal)
    check(-grenze, funktion, dezimal)

def Fehler(funktion, x):
    return abs((2/3-funktion(x))/(2/3))
