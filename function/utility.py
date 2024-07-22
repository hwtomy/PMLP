import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cvxpy as cp
from scipy.stats import rankdata

"""
Mutual infortion for utility function
The input is the possibility of current Py, Px and the lamda in from V
"""
def mutual(lambda_star_j, PY_j, Px):
    ep = 1e-6#  avoid log 0
    # for i in range(len(lambda_star_j)):
    #     if lambda_star_j[i] == 0:
    #         Px[i] = ep
    return cp.multiply(PY_j, np.sum(np.multiply(np.log(lambda_star_j+(1e-9)), lambda_star_j @ Px)))# avoid zeros  for log, not good.

"""
Kai Square for utility function
The input is the possibility of current Py, Px and the lamda in from V
"""
def X_squared(lambda_star_j, PY_j, Px):
    return cp.multiply(PY_j, np.sum(np.multiply((lambda_star_j**2-1), Px)))


"""
TV for utility function
The input is the possibility of current Py, Px and the lamda in from 
"""
def TV (lambda_star_j, PY_j, Px):
    return cp.multiply(PY_j, np.sum(np.multiply(abs(lambda_star_j-1), 0.5*Px)))

"""
information gain for utility function
"""
def infgain(lambda_star_j, PY_j, Px):
    H_Y = -cp.sum(cp.multiply(PY_j, cp.log(PY_j)))

    H_XY = -cp.sum(cp.multiply(lambda_star_j * PY_j * Px, cp.log(lambda_star_j * PY_j * Px)))

    information_gain = H_Y - H_XY
    return cp.sum(information_gain)

"""
KD divergence for utility function
"""
def kldiv(Q, P, Px):
    return cp.multiply(P, cp.log(P) - cp.log(Q))


def spearman_rank_correlation(Py, lambda_star):
    rank_Py = rankdata(Py)
    rank_lambda_star = rankdata(lambda_star)
    rank_diff = rank_Py - rank_lambda_star
    n = len(rank_Py)
    return 1 - (6 * np.sum(rank_diff ** 2) / (n * (n ** 2 - 1)))

"""
Get post possibilty of Py_x after getting Py
"""
def  postpossibility(Px, Py, V):
    Ny = len(Py)
    Nx = len(Px)
    Pyr = np.zeros(Ny)
    Py_x = np.zeros((Nx, Nx))
    j = 0 # set a counter, as the number of non zeros Y is not sure
    for i in range(Ny):
        if Py[i] != 0:
            Pyr[j] = Py[i]
            for k in range(Nx):
                Py_x[j, k] = V[i, k] * Py[i] # get the post
            j = j + 1
    print(Py_x)
    print(j)
    return Py_x

"""
Calculation of region k if known eplison
"""
def kregion(Px, epilison):
    N = len(Px)
    for i in range(N):
           temp1 = np.sum(Px[0:N-i]) # K-1
           temp2 = np.sum(Px[0:N-i-1]) #k
           if epilison >= -np.log(temp1) and epilison < -np.log(temp2):
               return i






