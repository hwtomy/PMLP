import numpy as np


"""
Mutual information for evaluation
The input is the possibility of X, Y, the V matrix and the size of alphabet X
"""
def mutual(Px, Py, V, Nx):
    #initialize
    Ny = len(Py)
    Pyr = np.zeros(Ny)
    Px_y = np.zeros((Nx, Nx))
    j = 0
    #Calculate the P(x,y)
    for i in range(Ny):
        if Py[i] != 0:
            Pyr[j] = Py[i]
            for k in range(Nx):
                Px_y[j, k] = V[i,k]*Px[k]*Py[i]
            j = j + 1
    ep =1e-9
    mutual = 0
    for i in range(Nx):
        for j in range(Nx):
                temp = (Nx*Px_y[i, j]) / (Px[j] * Pyr[i])
                if temp ==0:
                    temp = ep #avoid log(0)
                #print (temp)
                mutual += Px_y[i, j]/Nx * np.log(temp)#Get the  mutual information
    return mutual
"""
This the the function for ?
"""
def coefficient(X, Y):
    xmean = np.mean(X)
    ymean = np.mean(Y)
    r = np.sum((X - xmean) * (Y - ymean)) / np.sqrt(np.sum((X - xmean)**2) * np.sum((Y - ymean)**2))
    return r


"""
Evaluation  based on Pearson correlation
Input is the possibility of X, Y and the size of alphabet X
"""
def pearson(Px, Py, Nx):
    print(sum(Px))
    data_points = np.arange(Nx)
    j = 0
    le = 10000 # set the sample size
    Ny = len(Py)
    Pyr = np.zeros(Nx)
    for i in range(Ny):
        if Py[i] != 0:
            Pyr[j] = Py[i]
            j = j + 1
    print(sum(Px))

    ##generate the samples
    X_sample = np.random.choice(data_points, size=le, p=Px)
    Y_sample = np.random.choice(data_points, size=le, p=Pyr)


    X_mean = np.mean(X_sample)
    Y_mean = np.mean(Y_sample)

    #personal coefficient
    numerator = np.sum((X_sample - X_mean) * (Y_sample - Y_mean))
    denominator = np.sqrt(np.sum((X_sample - X_mean) ** 2) * np.sum((Y_sample - Y_mean) ** 2))
    pearson_correlation = numerator / denominator

    #explained variance
    # numerator = cp.sum((X_sample - Y_sample)**2)
    # denominator = cp.sqrt(cp.sum((X_sample - X_mean)**2)
    #
    # pearson_correlation = 1-numerator / denominator

    #spearman's corrletation

    return pearson_correlation

