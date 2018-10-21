import funktionen as funkt

#empirsiche Numerische Probe der beiden Funktionen. Wird nach 3 gefunden Ergebnissen (+dem 0 Fall abgebrochen).
funkt.check(100000000, funkt.f, 0, 3)
funkt.check(1, funkt.g, 8, 2)
#print(abs(1-funkt.g(10**(-20))*1.5)*100, 2/3, funkt.g(10^(-20)))
