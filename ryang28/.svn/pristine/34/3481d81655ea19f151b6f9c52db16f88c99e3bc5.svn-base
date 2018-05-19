from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
dataset = []
input_file = open('data/data/iris.data', 'r')
for idx, row in enumerate(input_file):
    if(idx == 0):
        continue
    row = row[:-1]
    row = row.split(",")
    line = []
    line.append(float(row[0]))
    line.append(float(row[1]))
    line.append(float(row[2]))
    line.append(float(row[3]))
    line.append(row[4])
    dataset.append(line)
#print(dataset[0][0])
# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)
#print("Result Centers")
iter_count = 0
#Euclidean dist
def distance(v, c):
    return np.sum((v-c)**2)

def nearest_cluster(v, c0, c1, c2):
	dists = []
	dists.append(distance(v, c0))
	dists.append(distance(v, c1))
	dists.append(distance(v, c2))
	return np.argmin(np.array(dists))

def k_means(C):
    # Write your code here!
    global iter_count
    iter_count+=1
    C = np.array(C)
    #assignemnt
    cluster_0 = []
    cluster_1 = []
    cluster_2 = []

    for data in dataset:
    	value = data[:-1]
    	idx = nearest_cluster(value, C[0], C[1], C[2])
    	if idx == 0:
    		cluster_0.append(value)
    	elif idx == 1:
    		cluster_1.append(value)
    	else:
    		cluster_2.append(value)

    #update clusters
    C_new_0 = np.mean(np.array(cluster_0), axis = 0)
    C_new_1 = np.mean(np.array(cluster_1), axis = 0)
    C_new_2 = np.mean(np.array(cluster_2), axis = 0)
    C_new = []
    C_new.append(C_new_0)
    C_new.append(C_new_1)
    C_new.append(C_new_2)
    C_new = np.array(C_new)
    sum_norm = np.linalg.norm(C[0]-C_new_0)+np.linalg.norm(C[1]-C_new_1)+np.linalg.norm(C[2]-C_new_2)

    if sum_norm < 10**(-3):
        C_final = C_new
    else:
        C_final = k_means(C_new)

    return C_final

res = k_means(C)
#print("number of loops: ",iter_count)
print(res)
