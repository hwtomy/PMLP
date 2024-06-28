import numpy as np
import pandas as pd


def probp(pX, x):
    px = []
    for i in range(len(x)):
        px_i = pX(x[i])
        px.append(px_i)
    px = np.array(px)
    return px

def makelist(x):
    counts = x.value_counts()
    Xlist = counts.index
    Xlist = Xlist.tolist()
    Xlist = np.array(Xlist)
    Px = counts.values
    Px = Px / np.sum(Px)

    return Xlist, Px


