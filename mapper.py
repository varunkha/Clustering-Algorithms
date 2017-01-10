#!/usr/bin/env python
#Mapper
import numpy as np
import random
import scipy.spatial.distance as distance
import sys

data = np.genfromtxt(sys.stdin)
#data_id = data[:,0]
#data_truth = data[:,1]
data_genes = data[:,2:]

centroids = np.genfromtxt('centroids.txt')
n_clusters = centroids.shape[0]

centroid_distance = distance.cdist(data_genes, centroids, 'euclidean')
cluster_assignment = np.argmin(centroid_distance,axis=1)
cluster_assignment = cluster_assignment.reshape(data.shape[0],1)
data = np.append(cluster_assignment,data,axis=1)
#for i in range(n_clusters):
#	cluster = data_genes[cluster_assignment==i]
#	centroids[i] = cluster.sum(axis=0)
np.savetxt(sys.stdout, data, delimiter='\t')
