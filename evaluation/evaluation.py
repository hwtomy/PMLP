import numpy as np

def mutual(Px, Py, V, Nx):
    Ny = len(Py)
    Pyr = np.zeros(Ny)
    Px_y = np.zeros((Nx, Nx))
    j = 0
    for i in range(Ny):
        if Py[i] != 0:
            Pyr[j] = Py[i]
            for k in range(Nx):
                Px_y[j, k] = V[i,k]*Px[k]*Px[k]
            j = j + 1
    ep =1e-9
    mutual = 0
    for i in range(Nx):
        for j in range(Nx):
                temp = (Nx*Px_y[i, j]) / (Px[j] * Pyr[i])
                if temp ==0:
                    temp = ep
                #print (temp)
                mutual += Px_y[i, j]/Nx * np.log(temp)
    return mutual

def coefficient(X, Y):
    xmean = np.mean(X)
    ymean = np.mean(Y)
    r = np.sum((X - xmean) * (Y - ymean)) / np.sqrt(np.sum((X - xmean)**2) * np.sum((Y - ymean)**2))
    return r

def pearson(Px, Py, Nx):
    print(sum(Px))
    data_points = np.arange(Nx)
    j = 0
    le = 10000
    Ny = len(Py)
    Pyr = np.zeros(Nx)
    for i in range(Ny):
        if Py[i] != 0:
            Pyr[j] = Py[i]
            j = j + 1
    print(sum(Px))
    X_sample = np.random.choice(data_points, size=le, p=Px)
    #data_points1 = np.arange(len(Py))
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

