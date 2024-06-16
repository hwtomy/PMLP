import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt







def Vertex_numeration(epsilon, Px, N):
    k = len(epsilon)
    V = []
    for i in range(k):
        tsum = sum(Px[N-i,N])
        if tsum >=np.exp(-epsilon[i-1]):
            lambdav = np.zeros(N)
            for j in range(N-i, N):
                lambdav[j] = (1-np.exp(epsilon[i-1]))*(tsum - Px[j])/Px[j]
                for j1 in range(N-i, N):
                    if j1 != j:
                        lambdav[j1] = np.exp(epsilon[i-1])
                V.append(lambdav)
    return V


def mutual_information(V, Px):

    Py_x = V * Px
    Py = np.sum(Py_x)
    if Py == 0:
        return 0
    Px_y = Py_x / Py
    return np.sum(Py_x * np.log(Px_y / Px))


def optimal_privacy_mechanism(V, Px):

    num_vertices = len(V)
    dim = len(V[0])

    c = np.array([-mutual_information(vertex, Px) for vertex in V])

    # Equality constraints: sum(P_Y) = 1 and sum(P_Y[j] * V[j][i]) = 1 for all i
    A_eq = np.zeros((dim + 1, num_vertices))
    b_eq = np.zeros(dim + 1)
    b_eq[0] = 1

    for j, vertex in enumerate(V):
        A_eq[0, j] = 1
        for i in range(dim):
            A_eq[i + 1, j] = vertex[i]
            b_eq[i + 1] = 1

    # Bounds: 0 <= P_Y[j] <= 1 for all j
    bounds = [(0, 1) for _ in range(num_vertices)]

    # Solve the linear programming problem
    result = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        return result.x
    else:
        raise ValueError("No result")
