import pandas as pd
import numpy as np
from scipy.linalg import eig
import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg
useEgn = True
## read file
dir_ = "graph.csv"
graph = pd.read_csv(dir_, header = None, sep = ',').to_numpy()
print(graph)
## show how to work @
# nCol = 5
# zarib1 = 2
# zarib2 =2
# sum_ = (zarib1*zarib2) * nCol
# a = np.ones([3,nCol])*zarib1
# b = np.ones([nCol,1])*zarib2
# c = a @ b
# print(c)

def pagerank_power(A, max_iter=1000,tol=1e-06):

    n, _ = A.shape
    x = np.asarray(A.sum(axis=1)).reshape(-1)
    k = x.nonzero()[0]
    D_1 = sprs.csr_matrix((1 / x[k], (k, k)), shape=(n, n))
    z_T = (((x != 0) + (x == 0)) / n)[sp.newaxis, :]
    M =  A.T @ D_1

    r = np.ones([n,1])
    r = r / n
    oldx = np.zeros((n, 1))

    iteration = 0
    while sp.linalg.norm(r - oldx) > tol:
        oldx = r
        ## OLD WAY
        # x = M @ x + s @ (z_T @ x)
        # x = M @ x + (z_T @ x)
        # x = np.dot(M, x) + np.dot(s , np.dot(z_T , x))
        r = np.dot(M, r) + np.dot(z_T, r)
        iteration += 1
        if iteration >= max_iter:
            break

    r = r / sum(r)

    return r.reshape(-1)

print(pagerank_power(graph))

