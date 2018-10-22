import numpy as np
import time
import datetime

#Funktion f der Aufgabe
def f(x):
    return (x**3+1/3)-(x**3-1/3)

#Funktion g der Aufgabe
def g(x):
    #Damit bei x=0 kein Fehler geworfen wird, wird bei x=0 der Wert 2/3 zurückgegeben
    try:
        return ((3+x**3/3)-(3-x**3/3))/x**3
    except ZeroDivisionError:
        return 2/3

#Funktion g der Aufgabe, jedoch ohne Exception
def g2(x):
    return ((3+x**3/3)-(3-x**3/3))/x**3

#Funktion zum empirschen Testen der Funktionen. Es wird angegeben, welche Funktionen bis wohin (grenze) getestet werden soll, wie viele verschiedene Abweichungen erfasst und auf wie viele Dezimalstellen überprüft werden solls.
def check(grenze, funktion, dezimal, abweichungen):
    #Misst die benötigte Zeit bzw startet die Uhr zur Messung
    start = time.clock()
    datalog=open("datalog.txt", "a")
    datalog.write(str(datetime.datetime.now()))
    datalog.write(": \n \n")
    #Negative und nicht ganzzahlige Dezimalstellen werden hier abgefangen
    if dezimal <0 or dezimal%1 !=0:
        print("Die Anzahl der Dezimalstellen muss größer als 0 sein!")
        return 0
    Wert=[2/3]*(abweichungen+1)
    gefunden=0
    string="Fuer ein x von 0 bis "+ str(grenze)+ " mit " +str( dezimal)+ " Dezimalstellen ergeben sich folgende Abweichungen: \n"
    datalog.write(string)
    #Hier wird der Durclauf gestartet. Die Range wird mit den Dezimalzahlen angepasst, da z.B. ein Testdurchlauf von 0 bis 100 mit einer Dezimalstelle 1000 Durchläufe braucht und nicht 100.
    for x in range(abs(grenze)*10**dezimal):
        #wenn eine negative Grenze übergeben wurde, muss natürlich auch mit negativen x gerechnet werden
        if grenze < 0:
            x=-x

        #Wenn eine Abweichungen  von 1% gefunden und die gewünschte Zahl von verschieden hohen Abweichungen noch nicht gefunden wurde....
        if Fehler(funktion, x*10**(-dezimal))>0.01 and gefunden!=abweichungen and Wert[gefunden] != funktion(x*10**(-dezimal)):
            #....wird der Wert mit dem letzten Wert abgeglichen und ggf ausgegeben
            string="x=" + str(x*10**(-dezimal)) + "f(x)=" + str( funktion(x*10**(-dezimal)) ) +", Fehler : " +str( Fehler(funktion,x*10**(-dezimal))*100)+ "% \n"
            datalog.write(string)
            gefunden+=1
            Wert[gefunden]=funktion(x*10**(-dezimal))

        #Wenn die gewünschte Anzahl von Abweichungen gefunden wurde, wird überprüft, ob die Nullstelle schon dabei war oder nicht. Wird leider jedes mal kontrolliertself.
        if gefunden==abweichungen and len(Wert)>np.count_nonzero(Wert) :
                #wenn nicht: kommentarlos beenden
                break

        elif funktion(x*10**(-dezimal))==0 and gefunden == abweichungen:
            #wenn doch: Ausgabe
            datalog.write("\nZusaetzlich ergibt sich als Nullstelle: \n")
            string="x=" + str( x*10**(-dezimal)) + "f(x)=" + str( funktion(x*10**(-dezimal))) + ", Fehler : " + str( Fehler(funktion, x*10**(-dezimal))*100) + "% \n"
            datalog.write(string)
            break


    string="\nUnterschiedliche Abweichungen gesucht: " + str(abweichungen) + " \nUnterschiedliche Abweichungen gefunden: " + str(gefunden) +"\n"
    datalog.write(string)
    #gibt Zeit aus
    string="Dauer= " +str( time.clock()-start) + " s \n \n \n"
    datalog.write(string)
#Führt check von -x bis x aus.
def doublecheck(grenze, funktion, dezimal):
    check(grenze, funktion, dezimal)
    check(-grenze, funktion, dezimal)

#Berechnet die Abweichung
def Fehler(funktion, x):
    return abs(1-funktion(x)*1.5)
