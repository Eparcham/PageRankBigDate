import pandas as pd
import numpy as np
from scipy.linalg import eig

useEgn = True
## read file
dir_ = "graph.csv"
graph = pd.read_csv(dir_, header = None, sep = ',').to_numpy()
print(graph)
b=np.linalg.det(graph)
print(b)

n = len(graph)
print("number of Node: ", n)

iter_num = 100000 ## step of walk
teleport_prob = 0.1 # for walk random in pages

## preprocessing (make graph like sparse matrix)
print("*"*50)
processed_graph = []
for i in range(n):
    processed_graph.append([])
    for j in range(n):
        if graph[i,j] ==1:
            processed_graph[-1].append(j)

print(processed_graph)
## call fun take one step from cur_pos and return the new pos
cur_pos = np.random.randint(n) ## initial pos of random walk
def one_step():
    if len(processed_graph[cur_pos]) == 0: # dead end than we must teleport
       return np.random.randint(n)

    if np.random.rand()<teleport_prob: # make teleport  we can use random walk prob here
        return np.random.randint(n)
    else:
        n_options = len(processed_graph[cur_pos]) ## give number of node we can go
        return processed_graph[cur_pos][np.random.randint(n_options)]


pagerank = np.zeros([1,n])
for i in range(iter_num):
    cur_pos = one_step()
    pagerank[0,cur_pos] +=1

print(pagerank)  ## show vist page
pagerank = pagerank / iter_num
print(pagerank)  ## show prob page

if useEgn :
    ## coputation of pagerank using eigenvectors
    def prob(a,b):
        if len(processed_graph[a])==0: ## dead end
            return 1/n  ## go from a to b just with teleport
        else:
            if b in processed_graph[a]: ## we can go fram a to b or graph[a,b]==1
                return  teleport_prob*(1/n) + (1 - teleport_prob) * 1/(len(processed_graph[a]))
            else:
                return teleport_prob * (1/n)  # we can go to b just with teleport and with teleport prob

    M = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            M[i,j] = prob(i,j)
    # print(M)
    print(np.sum(M,axis=1))

    ## CAL EIG
    eig_vals, eig_vecs  = eig(M,left =True, right=False)
    idx = np.argmax(eig_vals)
    fast_pagerank = eig_vecs[:,idx]/np.sum(eig_vecs[:,idx])
    print(np.abs(fast_pagerank))



