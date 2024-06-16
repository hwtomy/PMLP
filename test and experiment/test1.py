import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from v_numeration import Vertex_numeration,optimal_privacy_mechanism
from prob import probp



if __name__ == "__main__":
    Px = probp(pX, x)
    V = Vertex_numeration(epsilon, Px, N)
    Py = optimal_privacy_mechanism(V, Px)


