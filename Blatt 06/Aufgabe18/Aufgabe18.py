#Wahrscheinlichkeit für ein Spiel:
P_F=9/14
#Wahrscheinlichkeiten für die geg. Parameter:
P_Wind_Stark=6/14
P_Temp_Kalt=6/14
P_Aussicht_Sonnig=3/14
P_Feucht_Hoch=7/14
#Wahrscheinlichkeit für die geg. Parameter in F:
P_Wind_F=1/3
P_Temp_F=1/3
P_Feucht_F=1/3
P_Aussicht_F=2/9
#Gesamtwahrscheinlichkeit:
P_F_W=(P_Wind_F*P_Temp_F*P_Feucht_F*P_Aussicht_F)*P_F/(P_Wind_Stark*P_Temp_Kalt*P_Aussicht_Sonnig*P_Feucht_Hoch)
print(P_F_W)
