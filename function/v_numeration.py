import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cvxpy as cp
from function.utility import mutual, spearman_rank_correlation, infgain, kldiv
from itertools import combinations



def Vertex_numeration(k, Px, N):
    epsilon = np.sum(-np.log(Px[0:N-k]))
    V = []
    #d = 0
    for i in range(k):
        for J in combinations(range(N), N - i):
            #print(Px.shape)
            #print(J)
            tsum = sum(Px[m] for m in J)
            if tsum >=np.exp(-epsilon):
                lambdav = np.zeros(N)
                for j in J:
                    lambdav[j] = 1-np.exp(epsilon)*(tsum - Px[j])/Px[j]
                    for j1 in J:
                        if j1 != j:
                            #print(j1,j)
                            lambdav[j1] = np.exp(epsilon)
                    #print(lambdav)
                    V.append(lambdav.copy())
                    #d += 1
    #print(d)
    return V


# def mutual_information(V, Px):
#
#     Py_x = V * Px
#     Py = np.sum(Py_x)
#     if Py == 0:
#         return 0
#     Px_y = Py_x / Py
#     return np.sum(Py_x * np.log(Px_y / Px))


# def optimal_privacy_mechanism(V, Px):
#
#     num_vertices = len(V)
#     dim = len(V[0])
#
#     c = np.array([-mutual_information(vertex, Px) for vertex in V])
#
#     # Equality constraints: sum(P_Y) = 1 and sum(P_Y[j] * V[j][i]) = 1 for all i
#     A_eq = np.zeros((dim + 1, num_vertices))
#     b_eq = np.zeros(dim + 1)
#     b_eq[0] = 1
#
#     for j, vertex in enumerate(V):
#         A_eq[0, j] = 1
#         for i in range(dim):
#             A_eq[i + 1, j] = vertex[i]
#             b_eq[i + 1] = 1
#
#     # Bounds: 0 <= P_Y[j] <= 1 for all j
#     bounds = [(0, 1) for _ in range(num_vertices)]
#
#     # Solve the linear programming problem
#     result = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
#
#     if result.success:
#         return result.x
#     else:
#         raise ValueError("No result")

def mutual_opt(Px,V, N):
    N1 = V.shape[0]
    Py = cp.Variable(N1)
    Pya = cp.Variable(N1, boolean=True)
    # Pxy = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #           Pxy[i,j] = V[i,j]* Py[i]*Px[j]

    constraints = [
        # Pxy >= 0,
        # cp.sum(Pxy, axis=1) == Px,
        # cp.sum(Pxy, axis=0) == Py
    ]
    print(V.shape)
    #print(Py.shape)

    for i in range(N):
        constraints.append(cp.sum([Py[j] * V[j, i] for j in range(N1)]) == 1)

    objective = cp.Maximize(cp.sum([mutual(V[j, :], Py[j], Px) for j in range(N1)]))

    constraints += [
        cp.sum(Py) == 1,
        Py >= 0
    ]
    M = 1
    constraints += [Py[i] <= M * Pya[i] for i in range(N1)]
    constraints += [Py[i] >= 0 for i in range(N1)]

    proby = cp.Problem(objective, constraints)
    result = proby.solve(solver=cp.ECOS_BB)

    return Py.value

def person_opt(Px, V, N):
    Py = cp.Variable(N)
    Pxy = cp.Variable((N, N))

    constraints = [
        Pxy >= 0,
        cp.sum(Pxy, axis=1) == Px,  #
        cp.sum(Pxy, axis=0) == Py  #
    ]

    for i in range(N):
        constraints.append(cp.sum([Py[j] * V[i, j] for j in range(N)]) == 1)

    constraints += [
        cp.sum(Py) == 1,  #
        Py >= 0
    ]

    data_points = np.arange(len(Px))
    X_sample = np.random.choice(data_points, size=N, p=Px)
    #data_points1 = np.arange(len(Py))
    Y_sample = np.random.choice(data_points, size=N, p=Px)


    X_mean = np.mean(X_sample)
    Y_mean = np.mean(Y_sample)

    #personal coefficient
    numerator = cp.sum((X_sample - X_mean) * (Pxy - Y_mean))
    denominator = cp.sqrt(cp.sum((X_sample - X_mean) ** 2) * cp.sum((Pxy - Y_mean) ** 2))
    pearson_correlation = 1 - numerator / denominator

    #explained variance
    # numerator = cp.sum((X_sample - Y_sample)**2)
    # denominator = cp.sqrt(cp.sum((X_sample - X_mean)**2)
    #
    # pearson_correlation = 1-numerator / denominator

    #spearman's corrletation

    objective = cp.Maximize(cp.sum([Py[j] * pearson_correlation[j] for j in range(N)]))

    proby = cp.Problem(objective, constraints)
    result = proby.solve()

    return Py.value

# def infgain_opt(Px,V, N):
#     N1 = np.shape(V)[0]
#     Py = cp.Variable(N1)
#     Pxy = cp.Variable((N, N))
#     PXY = cp.multiply(V, Px.reshape(-1, 1)) * Py
#
#     constraints = [
#         Pxy >= 0,
#         cp.sum(Pxy, axis=1) == Px,
#         cp.sum(Pxy, axis=0) == Py
#     ]
#
#     for i in range(N):
#         constraints.append(cp.sum(cp.multiply(Py, V[i, :])) == 1)
#
#     H_Y = -cp.sum(cp.multiply(Py, cp.log(Py + 1e-9)))
#
#     #  H(Y|X)
#     H_Y_given_X = -cp.sum(cp.multiply(PXY, cp.log(PXY + 1e-9)))
#
#     # 计算信息增益 IG(X, Y)
#     information_gain = H_Y - H_Y_given_X
#
#     objective = cp.Maximize(cp.sum([Py[j] * mutual_information[j] for j in range(N)]))
#
#     constraints += [
#         cp.sum(Py) == 1,
#         Py >= 0
#     ]
#
#     proby = cp.Problem(objective, constraints)
#     result = proby.solve()
#
#     return Py.value

def spearman(Px, V, N):
    Py = cp.Variable(N)
    Pxy = cp.multiply(V, Px.reshape(-1, 1)) * Py

    constraints = [
        Pxy >= 0,
        cp.sum(Pxy, axis=1) == Px,
        cp.sum(Pxy, axis=0) == Py
    ]

    for i in range(N):
        constraints.append(cp.sum([Py[j] * V[i, j] for j in range(N)]) == 1)

    constraints += [
        cp.sum(Py) == 1,  #
        Py >= 0
    ]

    Py_initial = np.random.rand(N)
    Py_initial = Py_initial / np.sum(Py_initial)

    initial_spearman_terms = [spearman_rank_correlation(Py_initial, V[:, j]) for j in range(N)]
    initial_spearman_corr = np.sum(initial_spearman_terms)

    initial_objective = cp.Maximize(initial_spearman_corr)
    initial_prob = cp.Problem(initial_objective, constraints)
    initial_result = initial_prob.solve()

    Py_initial = Py.value

    # 最终目标函数
    spearman_terms = [spearman_rank_correlation(Py_initial, V[:, j]) for j in range(N)]
    spearman_corr = cp.sum(spearman_terms)

    # 定义最终目标函数，最大化正相关系数
    final_objective = cp.Maximize(spearman_corr)
    final_prob = cp.Problem(final_objective, constraints)
    final_result = final_prob.solve()

    return Py.value




