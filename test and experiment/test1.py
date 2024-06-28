import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from function.v_numeration import Vertex_numeration, mutual_opt, person_opt
from function.prob import makelist
import pandas as pd
import os



if __name__ == "__main__":
    file_path = 'F:\\PMLP\\data\\winequality-red-cleaned.csv'
    data = pd.read_csv(file_path)
    X = data.iloc[:, 0]
    Xlist, Px = makelist(X)
    N = len(Px)
    k = 2

    Px = np.array(Px[0:3])
    Px = Px/ np.sum(Px)
    #print(sum(Px))
    #print(Px)
    N = len(Px)
    V = Vertex_numeration(k, Px, N)
    V = np.array(V)
    print(V)
    Py = mutual_opt(Px, V, N)
    print(Py)

    # output_dir = 'result'
    #
    # output_file_path = os.path.join(output_dir, 'wineout.txt')
    #
    # with open(output_file_path, 'w') as f:
    #     # 写入V的标签和数据
    #     f.write("V\n")
    #     for item in V:
    #         f.write(f"{item}\n")
    #
    #     # 写入Py的标签和数据
    #     f.write("Py\n")
    #     for item in Py:
    #         f.write(f"{item}\n")





