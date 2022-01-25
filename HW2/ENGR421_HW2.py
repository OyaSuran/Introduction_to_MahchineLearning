#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np


# In[3]:


data_set = np.loadtxt("hw02_images.csv", delimiter = ",")

def safelog(x):
  return(np.log(x+1e-100))


# In[4]:


x_train = data_set[:30000,:]
x_test = data_set[30000:,:]


# In[6]:


y = np.loadtxt("hw02_labels.csv", delimiter = ",")


# In[8]:


y_train = y[:30000]
y_test = y[30000:]
K = np.max(y)


# In[57]:


training_images = data_set[:30000,:]


# In[9]:


print(K)


# In[45]:


sample_means = []

for c in range (1,6):
    sample_means.append(np.mean(x_train[y_train == c], axis = 0))


print(sample_means)


# In[91]:


sample_deviations =[]
for c in range (1,6):
    sample_deviations.append(np.sqrt(np.mean((x_train[y_train == c] - sample_means[c-1])**2, axis=0)))
print(sample_deviations)


# In[92]:


class_priors =[]

for c in range (1,6):
    class_priors.append(np.mean(y_train == c))

    
print(class_priors)


# In[93]:


score_values =[]
score_value1 = []

c=0
for i in range (0,30000):
    score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_train[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

score_values.append(score_value1)
score_value1 = []
c = c+1

for i in range (0,30000):
    score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_train[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

score_values.append(score_value1)
score_value1 = []
c = c+1

for i in range (0,30000):
    score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_train[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

score_values.append(score_value1)
score_value1 = []
c = c+1

for i in range (0,30000):
    score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_train[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

score_values.append(score_value1)
score_value1 = []
c = c+1

for i in range (0,30000):
    score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_train[i] )**2 /  sample_deviations[c]**2 )) +safelog(class_priors[c]))

score_values.append(score_value1)

print(np.shape(score_values))


# In[94]:


class_assignments = np.argmax(score_values, axis = 0)+1
print(class_assignments)
print(np.shape(class_assignments))


# In[95]:


confusion_matrix = pd.crosstab(class_assignments, y_train, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[96]:


test_score_values =[]
test_score_value1 = []

c=0
for i in range (0,5000):
    test_score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_test[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

test_score_values.append(test_score_value1)
test_score_value1 = []
c = c+1

for i in range (0,5000):
    test_score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_test[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

test_score_values.append(test_score_value1)
test_score_value1 = []
c = c+1

for i in range (0,5000):
    test_score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_test[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

test_score_values.append(test_score_value1)
test_score_value1 = []
c = c+1

for i in range (0,5000):
    test_score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_test[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

test_score_values.append(test_score_value1)
test_score_value1 = []
c = c+1

for i in range (0,5000):
    test_score_value1.append(np.sum((-0.5 * ( safelog(2 * math.pi * sample_deviations[c]**2 ))) + (-0.5 * (sample_means[c] - x_test[i] )**2 /  sample_deviations[c]**2 )) + safelog(class_priors[c]))

test_score_values.append(test_score_value1)

print(np.shape(test_score_values))


# In[97]:


test_class_assignments = np.argmax(test_score_values, axis = 0)+1
print(test_class_assignments)
print(np.shape(test_class_assignments))


# In[98]:


test_confusion_matrix = pd.crosstab(test_class_assignments, y_test, rownames = ['y_pred'], colnames = ['y_truth'])
print(test_confusion_matrix)

