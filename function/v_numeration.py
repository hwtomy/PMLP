import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cvxpy as cp
from function.utility import mutual, spearman_rank_correlation, infgain, kldiv, X_squared, TV
from itertools import combinations
import random
import gurobipy as gp
from gurobipy import GRB




def Vertex_numeration(k, Px, N):

    kf = -np.log(np.sum(Px[0:N-k+1]))
    kb = -np.log(np.sum(Px[0:N-k]))
    #print(kf,kb)
    #epsilon = random.uniform(kf,kb)
    #print(epsilon)
    epsilon = kf
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
                    lambdav[j] = (1-(np.exp(epsilon)*(tsum - Px[j])))/Px[j]
                    #print(lambdav[j])
                    for j1 in J:
                        if j1 != j:
                            #print(j1,j)
                            lambdav[j1] = np.exp(epsilon)
                    #print(lambdav)
                    V.append(lambdav.copy())
                    #d += 1
    #print(d)
    return V, epsilon


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
    #print(N1)
    Py = cp.Variable(N1)
    Pya = cp.Variable(N1, boolean=True)

    constraints = []
    #print(V.shape)
    #print(Py.shape)

    for i in range(N):
        constraints.append(cp.sum(cp.multiply(Py,V[:, i])) == 1)

    #objective = cp.Maximize(cp.sum([mutual(V[j, :], Py[j], Px) for j in range(N1)]))
    #objective = cp.Maximize(cp.sum([X_squared(V[j, :], Py[j], Px) for j in range(N1)]))
    objective = cp.Maximize(cp.sum([TV(V[j, :], Py[j], Px) for j in range(N1)]))

    constraints.append(cp.sum(Py) == 1)
    M = 1e6

    for i in range(N1):
        constraints.append(Py[i] <= M * Pya[i])
        constraints.append(Py[i] >= 0)

    #constraints += [Py[i] <= M * Pya[i] for i in range(N1)]
    constraints.append(cp.sum(Pya) <= N)
    #constraints += [Py[i] >= 0 for i in range(N1)]

    # print("Objective:", objective)
    # print("Constraints:")
    # for constraint in constraints:
    #     print(constraint)
    proby = cp.Problem(objective, constraints)
    #result = proby.solve(solver=cp.GUROBI, verbose=True, reoptimize=True,**{"Presolve": 0})
    result = proby.solve(solver=cp.GUROBI)
    # print("Problem status:", proby.status)

    # problem = cp.Problem(objective, constraints)
    #
    # # 自定义回调函数
    # def callback(model, where):
    #     if where == GRB.Callback.MIP:
    #         obj = model.cbGet(GRB.Callback.MIP_OBJBST)
    #         obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
    #         print(f"Current Best Objective: {obj}, Current Best Bound: {obj_bound}")
    #
    # try:
    #     # 将问题转换为 Gurobi 模型
    #     model = gp.Model()
    #
    #     # 添加变量
    #     gurobi_x = model.addMVar(shape=N1, vtype=GRB.CONTINUOUS, name="Py")
    #     gurobi_z = model.addMVar(shape=N1, vtype=GRB.BINARY, name="Pya")
    #
    #     # 添加约束
    #     for i in range(N1):
    #         model.addConstr(gurobi_x[i] <= M * gurobi_z[i], name=f"x_{i}_constr")
    #         model.addConstr(gurobi_x[i] >= 0, name=f"x_{i}_non_neg")
    #
    #     model.addConstr(gurobi_x.sum() == 1, name="sum_1")
    #     model.addConstr(gurobi_z.sum() <= N, name="non_zero")
    #
    #     # 设置目标函数
    #     model.setObjective(gurobi_x.sum(), GRB.MAXIMIZE)
    #
    #     # 设置回调函数
    #     model.optimize(callback)
    #
    #     # 输出结果
    #     print("Optimal value:", model.ObjVal)
    #     print("Optimal solution x:", gurobi_x.X)
    #     print("Non-zero elements in x:", np.count_nonzero(gurobi_x.X))
    # except gp.GurobiError as e:
    #     print("Gurobi error:", e)
    #
    # # 检查问题状态
    # status = model.Status
    # if status == GRB.INFEASIBLE:
    #     print("The problem is infeasible.")
    # elif status == GRB.UNBOUNDED:
    #     print("The problem is unbounded.")
    # elif status == GRB.OPTIMAL:
    #     print("The problem is optimal.")
    # else:
    #     print("Solver terminated with status:", status)

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




