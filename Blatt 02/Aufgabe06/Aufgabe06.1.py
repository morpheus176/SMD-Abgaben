import numpy as np
import matplotlib.pyplot as plt

def randomgenerator(a, b, m, x0, grenze, array, index):
    if int(grenze)==0:
        array = array[:-1]
        findperiod(array)
        print(array)
    else:
        xn=(a*x0+b) % m
        array[index]=xn
        array+=[None]
        #print(xn)
        return randomgenerator(a, b, m, xn, grenze-1, array, index+1)

def random(a, b, m, x0, grenze):
    list=[None]
    index=0
    randomgenerator(a, b, m, x0, grenze, list, index)

def findperiod(array):
    for x in range(len(array)-1):
        if array[0] == array[x+1]:
            print("Periodenlaenge: ", x+1)
            break

random(1, 3 , 1024, 0, 1000)
