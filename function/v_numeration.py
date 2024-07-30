import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import cvxpy as cp
from function.utility import mutual, spearman_rank_correlation, infgain, kldiv, X_squared, TV
from itertools import combinations
import random
import gurobipy as gp
from gurobipy import GRB



"""
This is the function to calculation Px|y/Px according to algorithm 1
It is now need input region k, Px, alphabet size N(may not need). If epsilon is input, k need to be calculate.
"""
def Vertex_numeration(k1, Px, N):

    #kf = -np.log(np.sum(Px[0:N-k+1]))#Get the lower bound of epsilon
    #kb = -np.log(np.sum(Px[0:N-k]))#Get the upper bound of epsilon
    #print(kf,kb)
    #epsilon = random.uniform(kf,kb)
    #print(epsilon)
    k = 3
    epsilon = np.log(3)
    V = []
    #d = 0
    for i in range(k):
        for J in combinations(range(N), N - i):##enumerate all the combination of N choose N-i
            #print(Px.shape)
            #print(J)
            tsum = np.sum(Px[m] for m in J)
            if tsum >=np.exp(-epsilon):
                lambdav = np.zeros(N)
                for j in J:
                    lambdav[j] = (1-(np.exp(epsilon)*(tsum - Px[j])))/Px[j]
                    if lambdav[j] >= 0: #avoid negative value
                        #print(lambdav[j])
                        for j1 in J:
                            if j1 != j:
                                #print(j1,j)
                                lambdav[j1] = np.exp(epsilon)
                        #print(lambdav)
                        V.append(lambdav.copy())#get possible Px_y/Px
                    #d += 1
    print(V)
    return V, epsilon

"""
This is the function to calculation Px|y/Px according to algorithm 1
It is now need input Px V matrix got from Vertex_numeration. 
Different utility function can be chosen.
Different solver can be chosen if the solver cen be used to solve MIX question.
"""
def mutual_opt(Px,V, N):
    N1 = V.shape[0]
    #print(N1)
    Py = cp.Variable(N1)
    Pya = cp.Variable(N1, boolean=True) # Control the number of non-zero elements in Py

    constraints = []
    #print(V.shape)
    #print(Py.shape)

    for i in range(N):
        constraints.append(cp.sum(cp.multiply(Py,V[:, i])) == 1) #The condition 2 in theorem 5

    ##Different utility function can be chosen in the maxinum question
    #objective = cp.Maximize(cp.sum([mutual(V[j, :], Py[j], Px) for j in range(N1)]))#mutual information
    #objective = cp.Maximize(cp.sum([X_squared(V[j, :], Py[j], Px) for j in range(N1)]))#Kai square
    objective = cp.Maximize(cp.sum([TV(V[j, :], Py[j], Px) for j in range(N1)]))# TV

    constraints.append(cp.sum(Py) == 1)
    M = 1e9# set a large number

    ## Control the number of nonzeros elements in Py
    for i in range(N1):
        constraints.append(Py[i] <= M * Pya[i])
        constraints.append(Py[i] >= -M * Pya[i])
        constraints.append(Py[i] >= 0)

    constraints.append(cp.sum(Pya) <= N) # Non zeros elements in Py less than size of Px
    #print(N)

    # print("Objective:", objective)
    # print("Constraints:")
    # for constraint in constraints:
    #     print(constraint)
    proby = cp.Problem(objective, constraints)

    # Below two solver is ok, others not gunrantee.
    #result = proby.solve(solver=cp.GUROBI)
    result = proby.solve(solver=cp.CBC)
    """
    This is a MIX solver problem, but not every solver that designed to solve MIX can be used.
    Some will get strange result, but it may not turn out to be wrong. Sorry for have forgotten the name of the solver.
    If want to use different other than the folowing two, please do some test process.
    """
    # print("Problem status:", proby.status)


    """
    The following part is uesed to see the process of solver. When confusing with the result come from the solver, this block should be used
    Bseides, it may contribte to time improvement as the iteration can be see in this block.
    """
    # problem = cp.Problem(objective, constraints)
    #
    # #
    # def callback(model, where):
    #     if where == GRB.Callback.MIP:
    #         obj = model.cbGet(GRB.Callback.MIP_OBJBST)
    #         obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
    #         print(f"Current Best Objective: {obj}, Current Best Bound: {obj_bound}")
    #
    # try:
    #     #  Gurobi
    #     model = gp.Model()
    #
    #     #
    #     gurobi_x = model.addMVar(shape=N1, vtype=GRB.CONTINUOUS, name="Py")
    #     gurobi_z = model.addMVar(shape=N1, vtype=GRB.BINARY, name="Pya")
    #
    #     #
    #     for i in range(N1):
    #         model.addConstr(gurobi_x[i] <= M * gurobi_z[i], name=f"x_{i}_constr")
    #         model.addConstr(gurobi_x[i] >= 0, name=f"x_{i}_non_neg")
    #
    #     model.addConstr(gurobi_x.sum() == 1, name="sum_1")
    #     model.addConstr(gurobi_z.sum() <= N, name="non_zero")
    #
    #     #
    #     model.setObjective(gurobi_x.sum(), GRB.MAXIMIZE)
    #
    #     #
    #     model.optimize(callback)
    #
    #     #
    #     print("Optimal value:", model.ObjVal)
    #     print("Optimal solution x:", gurobi_x.X)
    #     print("Non-zero elements in x:", np.count_nonzero(gurobi_x.X))
    # except gp.GurobiError as e:
    #     print("Gurobi error:", e)
    #
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

"""
This is use of information gain as utility function, but have not test yet.
"""
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
#     #  IG(X, Y)
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

"""
If calculate based on outcomes, this may work.
But indeed, it had better not to be used. To the information by now, it should be the same as other utility function if it is correct.
Just reserve here as a reference.
"""
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

    #
    spearman_terms = [spearman_rank_correlation(Py_initial, V[:, j]) for j in range(N)]
    spearman_corr = cp.sum(spearman_terms)

    #
    final_objective = cp.Maximize(spearman_corr)
    final_prob = cp.Problem(final_objective, constraints)
    final_result = final_prob.solve()

    return Py.value




