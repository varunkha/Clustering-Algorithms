import numpy as np
import random
import scipy.spatial.distance
import sys
import shlex
import subprocess
import datetime as d
import matplotlib.pyplot as plt
import sklearn.decomposition.pca as pca

def calScore():
    global rand
    global jaccard
    m11=0.0
    m00=0.0
    m01=0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if data_truth[i]==data_truth[j] and cluster_assignment[i]==cluster_assignment[j]:
                m11 += 1
            elif data_truth[i]!=data_truth[j] and cluster_assignment[i]!=cluster_assignment[j]:
                m00 += 1
            else:
                m01 += 1
    rand = (m11+m00)/(m11+m00+m01)
    jaccard = (m11)/(m11+m01)
	
n_clusters = 5
n_reducers = 1
file_name = 'cho.txt'

#initial random centroid assignment
iterations_output = 0
rand_output = 0
jaccard_output = 0
data = np.genfromtxt(file_name)
data_id = data[:,0]
data_truth = data[:,1]
data_genes = data[:,2:]
while iterations_output!=1:
    data = np.genfromtxt(file_name)
    data_id = data[:,0]
    data_truth = data[:,1]
    data_genes = data[:,2:]
    command_line = 'sh start.sh'
    args = shlex.split(command_line)
    p = subprocess.Popen(args)
    p.wait()
    iterations_output += 1
    centroids = np.asarray(random.sample(data_genes,  n_clusters))
    print centroids.shape
    centroids = np.sort(centroids)
    np.savetxt('centroids.txt', centroids, delimiter='\t')
    command_line = 'hadoop fs -copyFromLocal -f centroids.txt /'
    args = shlex.split(command_line)
    p = subprocess.Popen(args)
    p.wait()
    iterations = 0
    while True and iterations!=5:
        iterations += 1
        centroids_old = centroids.copy()
        command_line = 'hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar -file /home/varunhadoop/mapper.py -mapper /home/varunhadoop/mapper.py -file /home/varunhadoop/reducer.py -reducer /home/varunhadoop/reducer.py -input /cho.txt -output /centroids -numReduceTasks ' + str(n_reducers)
        args = shlex.split(command_line)
        p = subprocess.Popen(args)
        p.wait()

        command_line = 'sh file_process.sh'
        args = shlex.split(command_line)
        p = subprocess.Popen(args)
        p.wait()

        centroids = np.genfromtxt('centroids.txt')
        centroids = np.sort(centroids)
        if np.array_equal(centroids,centroids_old):
            break
    
    data = np.genfromtxt('data_assignment.txt')
    print data.shape
    data_id = data[:,1]
    data_truth = data[:,2]
    data_genes = data[:,3:]
    cluster_assignment = data[:,0]
    #print data_id.shape
    if(data_id.shape!=(386,)):
        exit()
    #print data_truth.shape
    #print data_genes.shape
    #print cluster_assignment.shape
    calScore()
    if rand > rand_output:
        rand_output = rand
        jaccard_output = jaccard
        cluster_assignment_output = cluster_assignment.copy()
        centroids_output = centroids.copy()
print "Rand Score:\t", rand_output
print "Jaccard Score:\t", jaccard_output

#Dimensionality reduction using PCA on data points
pca_d = pca.PCA(n_components=2)
pca_d.fit(data_genes)
pca_output = pca_d.transform(data_genes)

#Dimensionality reduction using PCA on centroids calculated using kmeans
pca_c = pca.PCA(n_components=2)
pca_c.fit(centroids_output)
pca_output_c = pca_c.transform(centroids_output)

#Dimensionality reduction using PCA on centroids calculated using ground truth
centroids_ground = np.zeros((5,16))
centroids_ground[0] = data_genes[data_truth==1].mean(axis=0)
centroids_ground[1] = data_genes[data_truth==2].mean(axis=0)
centroids_ground[2] = data_genes[data_truth==3].mean(axis=0)
centroids_ground[3] = data_genes[data_truth==4].mean(axis=0)
centroids_ground[4] = data_genes[data_truth==5].mean(axis=0)
pca_c_g = pca.PCA(n_components=2)
pca_c_g.fit(centroids_ground)
pca_output_c_g = pca_c_g.transform(centroids_ground)

#Plotting ground truth and kmeans clustering results
fig = plt.figure(num=None, figsize=(12, 6), dpi=96, facecolor='w', edgecolor='k')
plt.subplot(121)
plt.scatter(pca_output[data_truth==1][:,0],pca_output[data_truth==1][:,1],c='r',marker='+')
plt.scatter(pca_output[data_truth==2][:,0],pca_output[data_truth==2][:,1],c='b',marker='o')
plt.scatter(pca_output[data_truth==3][:,0],pca_output[data_truth==3][:,1],c='y',marker='s')
plt.scatter(pca_output[data_truth==4][:,0],pca_output[data_truth==4][:,1],c='k',marker='*')
plt.scatter(pca_output[data_truth==5][:,0],pca_output[data_truth==5][:,1],c='g',marker='^')
plt.scatter(pca_output_c_g[:,0],pca_output_c_g[:,1],c='m',marker='8',s=100)
plt.title('Ground Truth')

plt.subplot(122)
plt.scatter(pca_output[cluster_assignment==0][:,0],pca_output[cluster_assignment==0][:,1],c='y',marker='s')
plt.scatter(pca_output[cluster_assignment==1][:,0],pca_output[cluster_assignment==1][:,1],c='g',marker='^')
plt.scatter(pca_output[cluster_assignment==2][:,0],pca_output[cluster_assignment==2][:,1],c='b',marker='o')
plt.scatter(pca_output[cluster_assignment==3][:,0],pca_output[cluster_assignment==3][:,1],c='k',marker='*')
plt.scatter(pca_output[cluster_assignment==4][:,0],pca_output[cluster_assignment==4][:,1],c='r',marker='+')
plt.scatter(pca_output_c[:,0],pca_output_c[:,1],c='m',marker='8',s=100)
plt.title('Kmeans')
plt.show()
