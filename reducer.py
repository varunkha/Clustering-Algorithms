#!/usr/bin/env python
#Reducer
import numpy as np
import random
import scipy.spatial.distance
import sys

try:
    data = np.genfromtxt(sys.stdin)
    #data_id = data[:,1]
    #data_truth = data[:,2]
    if(data.ndim==1):
        data = data.reshape(1,data.shape[0])
    data_genes = data[:,3:]
    cluster_assignment = data[:,0]


    cluster_id = np.unique(cluster_assignment.flatten())
    n_clusters = cluster_id.shape[0]

    centroids = np.zeros((n_clusters,data_genes.shape[1]))

    for i in range(n_clusters):
        cluster = data_genes[cluster_assignment==cluster_id[i]]
        centroids[i] = cluster.mean(axis=0)
    np.savetxt(sys.stdout, centroids, delimiter='\t')
    print 'hello'
    np.savetxt(sys.stdout, data, delimiter='\t')
except IndexError:
    print 'hello'
#f=open('data.txt','a')
#np.savetxt(f,data)
#f.close()
