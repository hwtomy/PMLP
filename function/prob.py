import numpy as np
import pandas as pd


def probp(pX, x):
    px = []
    for i in range(len(x)):
        px_i = pX(x[i])
        px.append(px_i)
    px = np.array(px)
    return px

"""
This is the function for calculating the possibility of the dataset:From X get PX
The input is the dataset X
"""
def makelist(x):
    counts = x.value_counts()
    Xlist = counts.index
    Xlist = Xlist.tolist()
    Xlist = np.array(Xlist)
    Nx= counts.values
    Px =Nx / np.sum(x)

    return Xlist, Px, Nx


