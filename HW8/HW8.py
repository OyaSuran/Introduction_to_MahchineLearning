#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.spatial as spa


# In[2]:


X = np.genfromtxt("hw08_data_set.csv", delimiter=",", skip_header=1)
D = np.shape(X)[1]
N = np.shape(X)[0] 
K = 5
print("Number of data points:",N)
print("Number of features:",D)
print("Number of clusters:",K)


# In[3]:


plt.figure(figsize=(8, 8))
plt.plot(X[:,0],X[:,1], "k.", markersize = 10)

plt.xlabel("x1")
plt.ylabel("x2")


# In[4]:


matrixB = np.zeros((N,N))
sigma = 1.25

for i in range(N):
    for j in range(N):
        if i!=j:
            matrixB[i,j] = (spa.distance.euclidean(X[i],X[j]) <= sigma)

plt.figure(figsize=(8, 8))

for n in range(N):
    plt.plot(X[n, 0], X[n, 1], ".", markersize=10, c="k")

for i in range(N):
    for j in range(i, N):
        if matrixB[i, j] == 1.:
            x1, x2 = X[i, 0], X[j, 0]
            y1, y2 = X[i, 1], X[j, 1]
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.3)
            
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Connectivity Matrix Visualization")
plt.show()


# In[5]:


matrixD = np.zeros((N,N))
for i in range(N):
    matrixD[i,i] = np.sum(matrixB[i,:])

L = matrixD-matrixB

I = np.identity(N)
D_inv = np.linalg.inv(matrixD)

L_symmetric = I - np.sqrt(np.dot(np.dot(D_inv, matrixB), D_inv))

eigvals, eigvecs = np.linalg.eig(L_symmetric)
idx = np.argsort(eigvals)
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]
Z = eigvecs[:,:5]


# In[6]:


initial_centroids = np.zeros((K,K))

initial_centroids[0] = Z[29]
initial_centroids[4] = Z[143]
initial_centroids[1] = Z[204]
initial_centroids[2] = Z[271]
initial_centroids[3] = Z[277]


# In[12]:


print(Z[0])
print(Z[29])
print(Z[143])
print(Z[204])
print(Z[271])
print(Z[277])


# In[8]:


def centroids_update(memberships, M, initcent):
    if memberships is not None:
         centroids = np.vstack([np.mean(M[memberships == k,], axis = 0) for k in range(K)])
    else:
        centroids = initcent
    return(centroids)

def memberships__update(centroids, M):
    matrixD = spa.distance_matrix(centroids, M)
    memberships = np.argmin(matrixD, axis = 0)
    return(memberships)


# In[9]:


centroids = None
memberships = None
iteration = 1

while True:
    print("Iteration #{}:".format(iteration))

    oldcentroids = centroids
    centroids = centroids_update(memberships, Z, initial_centroids)
    print(centroids)
    if np.alltrue(centroids == oldcentroids):
        break

    oldmemberships = memberships
    memberships = memberships__update(centroids, Z)
   
    if np.alltrue(memberships == oldmemberships):
        break

    iteration = iteration + 1


# In[35]:


cluster_colors = np.array(["#1f78b4", "#e31a1c","#6a3d9a", "#ff7f00","#33a02c"])
plt.figure(figsize=(8, 8))

for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10, color=cluster_colors[c])

for c in range(K):
    total =0
    for i in range(0,len(X[memberships == c])-1):
        total = total + X[memberships == c][i]
    
    plt.plot((total/len(X[memberships == c]))[0], (total/len(X[memberships == c]))[1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Resulting Clusters")
plt.show()


# In[10]:


print(centroids)


# In[29]:


print(np.argmin(X[memberships == c]))
total =0 
for i in range(0,len(X[memberships == 1])-1):
    total = total+X[memberships == 1][i]

print("total)
print(X[memberships == c][27])
print(X[memberships == c])


# In[27]:


print(np.shape(X[memberships == 1]))
len(X[memberships == 1])

