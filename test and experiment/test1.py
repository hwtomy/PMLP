import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from function.v_numeration import Vertex_numeration, mutual_opt
from function.prob import makelist
import pandas as pd
from evaluation.evaluation import pearson, mutual
import datetime
import os
from function.utility import postpossibility



if __name__ == "__main__":
    file_path = 'F:\\PMLP\\data\\winequality-red-cleaned.csv'
    data = pd.read_csv(file_path)
    X = data.iloc[:, 0]
    Xlist, Px, Nx = makelist(X) #Get the possibility from dataset
    N = len(Px)
    k = 3
    #mutuals = np.zeros(26)
    #pearsons = np.zeros(26)
    #tests = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,25,30,40,50,60,70,80,90,95])
    #i = 0
    #for j in tests:
    # Px1=np.zeros(j)
    # Px1 = np.array(Px[0:j])
    # Px1 = Px1/ np.sum(Px1)
    #print(sum(Px1))
    #print(Px)
    Px1 = np.array([0.25, 0.25, 0.25, 0.25])
    N = len(Px1)
    Px1= np.sort(Px1)[::-1]#from large to small
    #print(sum(Px))
    V, ep = Vertex_numeration(k, Px1, N) #calculate the V matrix
    V = np.array(V)
    print(V)
    Py = mutual_opt(Px1, V, N) #calculate the Py
    Py_x = postpossibility(Px1, Py, V) #calculate the Py_x
    print(Py)
    #print(sum(Py))
    #print(sum(Px1))
    #print(Py_x)

        # Pearson1 = pearson(Px1, Py, N)
        #
        # mutual1 = mutual(Px1, Py,V, N)
        # mutuals[i] = mutual1
        # pearsons[i] = Pearson1
        # i = i + 1
        ##Write the result to the file, save in result in test and experiment
        # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # save_str = "k={}_N={}".format(k, j)
        # with open('result/%s_log_%s.txt' % (save_str, timestamp), 'w') as f:
        #     f.write(timestamp)
        #     f.write('\n')
        ## write the V matrix
        #     f.write("V:\n")
        #     for row in V:
        #         row_str = ', '.join(f'{num:.6f}' for num in row)
        #         f.write(row_str + '\n')
        #     f.write('\n')
        ## write the Py
        #     f.write("Py:\n")
        #     array_str = ', '.join(f'{num:.6f}' for num in Py)
        #     f.write(array_str + '\n')
        ## write the mutual information
        #     f.write("Mutual_information: {}\n".format(mutual1))
        #
        #     f.write("Pearson_coefficient: {}\n".format(Pearson1))
    #print(mutuals)
    #print(pearsons)
    ##Save the evaluation as csv for further analysis
    #df = pd.DataFrame(mutuals, columns=['Value'])
    #df.to_csv('vertical_data.csv', index=False)
    #df = pd.DataFrame(pearsons, columns=['Value'])
    #df.to_csv('vertical_data1.csv', index=False)




