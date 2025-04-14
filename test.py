import numpy as np


points1 = np.array([[0,1],[1,0]])

pearson = np.corrcoef(points1.T)[0,1]
print("pearson", pearson)