import numpy as np
import time
import matplotlib.pyplot as plt
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

#Das Argument der zu untersuchenden Funktion für die Fälle StartwertYGrenze und umgekehrt. Wurde hier definiert um Korrekturen einfacher zu machen.
def argument(x, dezimal, start, grenze):
    if start < grenze:
        return x*10**(-dezimal)+start
    else:
        return -x*10**(-dezimal)+start


#Funktion zum empirschen Testen der Funktionen. Es wird angegeben, welche Funktionen bis wohin (grenze) getestet werden soll, wie viele verschiedene Abweichungen erfasst und auf wie viele Dezimalstellen überprüft werden solls.
#Funktioniert für positive start und grenzwerte, negative nicht getestet
def check(start, grenze, funktion, dezimal, abweichungen):
    #Misst die benötigte Zeit bzw startet die Uhr zur Messung
    start_t = time.clock()
    #Öffnen einer Textdatei zum Speichern der Ergebnisse. Das, was in den log geschrieben wird, wird vorher in "string" gespeichertund dann mit datalog.write in die Datei geschrieben. Kenne leider keinen schöneren Weg.
    datalog=open("datalog.txt", "a")
    datalog.write(str(datetime.datetime.now()))
    datalog.write(": \n \n")
    #Negative und nicht ganzzahlige Dezimalstellen werden hier abgefangen
    if dezimal <0 or dezimal%1 !=0:
        print("Die Anzahl der Dezimalstellen muss größer als 0 und eine ganze Zahl sein!")
        return 0
    #Erstellung eines Array in dem Werte gespeichert werden
    Wert=[2/3]*(abweichungen+1)
    gefunden=0
    string="Fuer ein x von " + str(start) + " bis "+ str(grenze)+ " mit " +str( dezimal)+ " Dezimalstellen ergeben sich folgende Abweichungen: \n"
    datalog.write(string)
    #Hier wird der Durclauf gestartet. Die Range wird mit den Dezimalzahlen angepasst, da z.B. ein Testdurchlauf von 0 bis 100 mit einer Dezimalstelle 1000 Durchläufe braucht und nicht 100.
    for x in range(abs((max(start, grenze)+max(-start, -grenze))*10**dezimal)):

        #Wenn eine Abweichungen  von 1% gefunden und die gewünschte Zahl von verschieden hohen Abweichungen noch nicht gefunden wurde....
        if Fehler(funktion, argument(x, dezimal, start, grenze))>0.01 and gefunden!=abweichungen and Wert[gefunden] != funktion(argument(x, dezimal, start, grenze)):
            #....wird der Wert mit dem letzten Wert abgeglichen und ggf ausgegeben
            string="x=" + str(argument(x, dezimal, start, grenze)) + "  f(x)=" + str( funktion(argument(x, dezimal, start, grenze)) ) +", Fehler : " +str( Fehler(funktion,argument(x, dezimal, start, grenze))*100)+ "% \n"
            datalog.write(string)
            gefunden+=1
            Wert[gefunden]=funktion(argument(x, dezimal, start, grenze))

        #Wenn die gewünschte Anzahl von Abweichungen gefunden wurde, wird überprüft, ob die Nullstelle schon dabei war oder nicht. Wird leider jedes mal kontrolliertself.
        if gefunden==abweichungen and len(Wert)>np.count_nonzero(Wert) :
                #wenn nicht: kommentarlos beenden
                break

        elif funktion(argument(x, dezimal, start, grenze))==0 and gefunden == abweichungen:
            #wenn doch: Ausgabe
            datalog.write("\nZusaetzlich ergibt sich als Nullstelle: \n")
            string="x=" + str( argument(x, dezimal, start, grenze)) + "f(x)=" + str( funktion(argument(x, dezimal, start, grenze))) + ", Fehler : " + str( Fehler(funktion, argument(x, dezimal, start, grenze))*100) + "% \n"
            datalog.write(string)
            break


    string="\nUnterschiedliche Abweichungen gesucht: " + str(abweichungen) + " \nUnterschiedliche Abweichungen gefunden: " + str(gefunden) +"\n"
    datalog.write(string)
    #gibt Zeit aus
    string="Dauer= " +str( time.clock()-start_t) + " s \n \n \n"
    datalog.write(string)

#Berechnet die Abweichung
def Fehler(funktion, x):
    return abs(1-funktion(x)*1.5)


#Lässt die Funktion check für f bzw g durchlaufen. f wird von 0-180000 kontrolliert, g von 1-0.
#Bei f werden nur ganzzalige x kontrolliert, bei g wird bis zur 8. Dezimalstelle gerechnet.
check(0, 180000, f, 0, 1)
check(1, 0, g, 8, 1)
#print (Fehler(g, 8.730000000012339e-06))

#
x=np.linspace(1e-7, 0.01, 500000)
x2=np.linspace(1e-10, 80*10**3, 500000)
y=Fehler(f, x2)*100
z=Fehler(g2, x)*100

# Der Fehler g wird von e-7 bis 0.01 geplottet und die x-Achse dabei logarithmiert.
plt.plot(np.log(x), z)
plt.xlabel(r'$\log{(x)}$')
plt.ylabel(r'Abweichung in %')

plt.savefig("Fehler1b.pdf")
plt.clf()

#Der Fehler von f wird von 0 bis 180000 geplottet.
plt.plot(x2*10**(-3), y)
plt.xlabel(r'$x \: \cdot \: 10^3$')
plt.ylabel('Abweichung in %')
plt.savefig("Fehler1a.pdf")
