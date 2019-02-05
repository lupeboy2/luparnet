import os
import numpy as np

def fit(csv):
    with open(csv, "r") as inputs:
        x=np.array()
        for line in inputs:
            for subline in line.split(','):
                x=np.append(x,[subline])
                
