import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cvxpy as cp


def Vertex_numeration(k, Px, N):
    epsilon = np.sum(-np.log(Px[0:N-k]))
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
    Py = cp.Variable(N)
    Pxy = cp.Variable((N, N))

    constraints = [
        Pxy >= 0,
        cp.sum(Pxy, axis=1) == Px,
        cp.sum(Pxy, axis=0) == Py
    ]

    for i in range(N):
        constraints.append(cp.sum(cp.multiply(Py, V[i, :])) == 1)

    H_X = -cp.sum(cp.multiply(Px, cp.log(Px)))
    H_Y = -cp.sum(cp.multiply(Py, cp.log(Py)))
    H_XY = -cp.sum(cp.multiply(Pxy, cp.log(Pxy)))

    mutual_information = H_X + H_Y - H_XY

    objective = cp.Maximize(mutual_information)

    constraints += [
        cp.sum(Py) == 1,
        Py >= 0
    ]

    proby = cp.Problem(objective, constraints)
    result = proby.solve()

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
        constraints.append(cp.sum(cp.multiply(Py, V[i, :])) == 1)

    data_points = np.arange(len(Px))
    X_sample = np.random.choice(data_points, size=N, p=Px)
    #data_points1 = np.arange(len(Py))
    Y_sample = np.random.choice(data_points, size=N, p=Px)

    # 计算均值
    X_mean = np.mean(X_sample)
    Y_mean = np.mean(Y_sample)

    # 定义皮尔逊相关系数的分子和分母部分
    numerator = cp.sum((X_sample - X_mean) * (Pxy - Y_mean))
    denominator = cp.sqrt(cp.sum((X_sample - X_mean) ** 2) * cp.sum((Pxy - Y_mean) ** 2))

    pearson_correlation = numerator / denominator

    objective = cp.Maximize(pearson_correlation)

    constraints += [
        cp.sum(Py) == 1,  #
        Py >= 0
    ]

    proby = cp.Problem(objective, constraints)
    result = proby.solve()

    return Py.value



