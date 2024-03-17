import numpy as np


def KNN(x, y, returnValues = 0):
    distances = []
    for axisX, axisY in zip(x, y):
        distance = axisX - axisY
        absDistance = np.absolute(distance)
        distances.append(absDistance)
        
    sortedDistances = []
    checkDistance = min(distances, key = lambda x:np.absolute(x-i)) 
    sortedDistances.append(checkDistance)
    distances.remove(checkDistance)
        
    if returnValues == 0:
        return sortedDistances[0]
    else:
        return sortedDistances[0:returnValues-1]
    
def RNN(w, u, b, x):
    yt = 0
    ht = 1 / 1 (w * x + u * yt ** -1 + b) ** -1
    yt = 1 / 1 (w * ht + b) ** -1

    return yt

