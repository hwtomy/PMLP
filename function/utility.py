import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cvxpy as cp
from scipy.stats import rankdata


def mutual(lambda_star_j, PY_j, Px):
    return cp.multiply(PY_j, cp.sum(cp.multiply(cp.log(lambda_star_j), lambda_star_j * Px)))

def infgain(lambda_star_j, PY_j, Px):
    H_Y = -cp.sum(cp.multiply(PY_j, cp.log(PY_j)))

    H_XY = -cp.sum(cp.multiply(lambda_star_j * PY_j * Px, cp.log(lambda_star_j * PY_j * Px)))

    information_gain = H_Y - H_XY
    return cp.sum(information_gain)

def kldiv(Q, P, Px):
    return cp.multiply(P, cp.log(P) - cp.log(Q))


def spearman_rank_correlation(Py, lambda_star):
    rank_Py = rankdata(Py)
    rank_lambda_star = rankdata(lambda_star)
    rank_diff = rank_Py - rank_lambda_star
    n = len(rank_Py)
    return 1 - (6 * np.sum(rank_diff ** 2) / (n * (n ** 2 - 1)))




