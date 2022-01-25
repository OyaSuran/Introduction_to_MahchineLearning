#!/usr/bin/env python
# coding: utf-8

# In[78]:


import math 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import scipy
from sklearn import svm


# In[79]:



# mean parameters
np.random.seed(521)
class1_mean = np.array([0.0, 2.5])
class2_mean = np.array([-2.5, -2.0])
class3_mean = np.array([2.5, -2.0])

class1_cov = [[3.2, 0.0], [0.0, 1.2]]
class2_cov = [[1.2, 0.8], [0.8, 1.2]]
class3_cov = [[1.2, -0.8], [-0.8, 1.2]]

class_means = np.array([class1_mean, class2_mean, class3_mean])
# standard deviation parameters
class_covs = np.array([class1_cov, class2_cov, class3_cov])
# sample sizes
class_sizes = np.array([120, 80, 100])




# In[80]:


# generate random samples
points1 = np.random.multivariate_normal(class_means[0], class_covs[0], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1], class_covs[1], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2], class_covs[2], class_sizes[2])
points = np.concatenate((points1, points2, points3))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))
X = np.vstack((points1,points2,points3))



# In[81]:



plt.figure(figsize = (6, 6))
# plot data points of the first class

plt.plot(points1[:,0],points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0],points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0],points3[:,1], "b.", markersize = 10)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[85]:



def find_mean(n,array):
    x=0
    y=0
    for i in range (0,n):
      x = x + array[i,0]
      y = y + array[i,1]
    x = (x/n)
    y = (y/n)
    return x,y

#print(find_mean(class_sizes[0],points1))



sample_means = [np.mean(points1, axis=0),np.mean(points2, axis=0),np.mean(points3, axis=0)]
print("Sample Means")
for i in range (0,3):
    print(sample_means[i])

def find_sample_cov(array):
    sij = np.cov(array[:,0],array[:,1])
    return sij


cov_array = []
#to gain true values for coveraiance we multiply by (n-1)/n
cov_array.insert(0,(find_sample_cov (points3))*99/100)
cov_array.insert(0,(find_sample_cov (points2))*79/80)
cov_array.insert(0,(find_sample_cov (points1))*119/120)

print("Sample Coveriances")
for i in range (0,3):
    print(cov_array[i])


class_priors = [(class_sizes[c]/sum(class_sizes)) for c in range(3)]
print("Class Priors")
print(class_priors)


# In[103]:


# To find socring(g) function we need Wc-> W, wc -> w, wc0 -> wc.

W = [((np.linalg.inv(cov_array[c]))*(-0.5)) for c in range(0,3)]

w1 = (np.matmul(np.linalg.inv(cov_array[0]),  (sample_means[0])))
w2 = (np.matmul(np.linalg.inv(cov_array[1]),  (sample_means[1])))
w3 = (np.matmul(np.linalg.inv(cov_array[2]),  (sample_means[2])))

w = [(np.matmul(np.linalg.inv(cov_array[c]),  (sample_means[c]))) for c in range(0,3)]


wc = [((-0.5 * np.matmul(np.matmul(np.transpose(sample_means[i]),(np.linalg.inv(cov_array[i]))),sample_means[i]))
       +(-0.5 * np.log(2*math.pi) * 2) + (-0.5 * np.log(np.linalg.det(cov_array[i]))) + np.log(class_priors[i])) for i in range(0,3)]

g1 =[]
g2 =[]
g3 =[]

#y = np.matmul((np.matmul(X, W[0])) , np.transpose(X)) + np.matmul(X, w[0]) + wc[0]

# then we find g score of each point I calculate it end appened it into g functions

for i in range (0,3):
    for x in range(0,300):
        J = np.matmul((np.matmul(points[x], W[i])) , np.transpose(points[x])) + np.matmul(points[x], w[i]) + wc[i]
        if(i== 0):
            g1.append(J)
        if(i== 1):
            g2.append(J)
        if(i== 2):
            g3.append(J)

g = np.stack([g1,g2,g3])


# In[104]:


class_assignments = np.argmax(g, axis = 0)

for i in range(0,300):
    if (class_assignments[i] == 2):
        class_assignments[i] = 3
        
    if (class_assignments[i] == 1):
        class_assignments[i] = 2
        
    if (class_assignments[i] == 0):
        class_assignments[i] = 1

y = np.concatenate((np.repeat(1,class_sizes[0]), np.repeat(2,class_sizes[1]),np.repeat(3,class_sizes[2])))

confusion_matrix = pd.crosstab(class_assignments, y, rownames= ["y_pred"], colnames=["y_truth"])
print("Confusion Matrix")
print(confusion_matrix)


# # Visualization

# In[111]:


x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)

plt.figure(figsize = (6, 6))


plt.plot(X[y == 1, 0], X[y == 1, 1], "r.", markersize = 10)
plt.plot(X[y == 2, 0], X[y == 2, 1], "g.", markersize = 10)
plt.plot(X[y == 3, 0], X[y == 3, 1], "b.", markersize = 10)

plt.plot(X[class_assignments != y, 0], X[class_assignments != y, 1], "ko", markersize = 12, fillstyle = "none")


#plt.contour(points1, points2, points3, levels = 0, colors = "k")
#plt.contour(x1_grid, x2_grid, discriminant_values, levels = 0, colors = "k")
X = np.concatenate((points1,points2, points3), axis = 0)
Y = np.array([1]*100 + [2]*100 + [3]*100)


clf = svm.SVC(kernel = 'rbf', C=C)
clf.fit(X, Y)

h = .02  

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1 

xmesh, ymesh = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


Z = clf.predict(np.c_[xmesh.ravel(), ymesh.ravel()])

Z = Z.reshape(xmesh.shape)
plt.contour(xmesh, ymesh, Z, cmap=plt.cm.Paired)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

