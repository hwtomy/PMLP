import numpy as np

def mutual(X, Y):
    xm = max(X)
    ym = max(Y)
    px = np.zeros(xm+1)
    py = np.zeros(ym+1)
    pxy = np.zeros((xm+1, ym+1))
    for i in range(len(X)):
        px[X[i]] += 1
        py[Y[i]] += 1
        pxy[X[i], Y[i]] += 1
    px = px / len(X)
    py = py / len(Y)
    pxy = pxy / len(X)
    mutual = 0
    N = len(X)
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] != 0 and px[i] != 0 and py[j] != 0:
                mutual += pxy[i, j]/N * np.log(N*pxy[i, j] / (px[i] * py[j]))
    return mutual

def coefficient(X, Y):
    xmean = np.mean(X)
    ymean = np.mean(Y)
    r = np.sum((X - xmean) * (Y - ymean)) / np.sqrt(np.sum((X - xmean)**2) * np.sum((Y - ymean)**2))
    return r

