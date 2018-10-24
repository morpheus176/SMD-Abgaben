import numpy as np

print("\n")

# a)/b)/c)/d)

rot = [1, 2, 3, 4, 5, 6]     # Werte, die der rote Würfel annehmen kann
blau = [1, 2, 3, 4, 5, 6]    # Werte, die der blaue Würfel annehmen kann

treffer_a = 0     # Variable, die die Anzahl der Treffer in a) zählen soll
treffer_b = 0     # Variable, die die Anzahl der Treffer in b) zählen soll
treffer_c = 0     # Variable, die die Anzahl der Treffer in c) zählen soll
treffer_d = 0     # Variable, die die Anzahl der Treffer in d) zählen soll

for xx in rot:
    for yy in blau:
        if yy + xx == 9:                                        # Berechnung der Treffer für a)
            treffer_a += 1
        if yy + xx >= 9:                                        # Berechnung der Treffer für b)
            treffer_b += 1
        if (xx == 4 and yy == 5) or (yy == 4 and xx == 5):      # Berechnung der Treffer für c)
            treffer_c += 1
        if xx == 4 and yy == 5:                                 # Berechnung der Treffer für d)
            treffer_d += 1

daten = [treffer_a, treffer_b, treffer_c, treffer_d]
texte = ["a) Wahrscheinlichkeit, dass die Summe der Punkte 9 ergibt:",
         "b) Wahrscheinlichkeit, dass die Summe der Punkte 9 oder mehr ergibt:",
         "c) Wahrscheinlichkeit, dass ein Wuerfel 4, der andere 5 Punkte zeigt:",
         "d) Wahrscheinlichkeit, dass der rote Wuerfel 4, der blaue 5 Punkte zeigt:"]

for data, text in zip(daten, texte):            # Berechnung und Ausgabe der Wahrscheinlichkeit für den jeweiligen Aufgabenteil
    data = data/(len(rot)*len(blau))*100
    print(text, "%0.2f %%" % data)

print("\n")

# e)/f)/g)

rot = [4]                       # Werte, die der rote Würfel annehmen kann
blau = [1, 2, 3, 4, 5, 6]       # Werte, die der blaue Würfel annehmen kann

treffer_e = 0     # Variable, die die Anzahl der Treffer in e) zählen soll
treffer_f = 0     # Variable, die die Anzahl der Treffer in f) zählen soll
treffer_g = 0     # Variable, die die Anzahl der Treffer in g) zählen soll

for xx in rot:
    for yy in blau:
        if yy + xx == 9:                # Berechnung der Treffer für e)
            treffer_e += 1
        if yy + xx >= 9:                # Berechnung der Treffer für f)
            treffer_f += 1
        if xx == 4 and yy == 5:         # Berechnung der Treffer für g)
            treffer_g += 1

daten = [treffer_e, treffer_f, treffer_g]
texte = ["e) Wahrscheinlichkeit, dass die Summe der Punkte 9 ergibt:",
         "f) Wahrscheinlichkeit, dass die Summe der Punkte 9 oder mehr ergibt:",
         "g) Wahrscheinlichkeit, dass ein Wuerfel 4, der andere 5 Punkte zeigt:"]

for data, text in zip(daten, texte):            # Berechnung und Ausgabe der Wahrscheinlichkeit für den jeweiligen Aufgabenteil
    data = data/(len(rot)*len(blau))*100
    print(text, "%0.2f %%" % data)
