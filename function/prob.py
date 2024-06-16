import numpy as np


def probp(pX, x):
    px = []
    for i in range(len(x)):
        px_i = pX(x[i])
        px.append(px_i)
    px = np.array(px)
    return px