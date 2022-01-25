#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def argsort(seq):
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]


# In[2]:


data_set_train = np.genfromtxt("hw05_data_set.csv", delimiter=",", skip_footer=122, skip_header=1)
data_set_test = np.genfromtxt("hw05_data_set.csv", delimiter=",", skip_header=151)


# In[3]:


x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1].astype(int)
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1].astype(int)


N_train = x_train.shape[0]
N_test = x_test.shape[0]
P = 25
#print("Number of samples (training):", N_train)
#print("Number of samples (test):", N_test)


# In[4]:


class BinaryTree:
    def __init__(self, x_train, y_train, p):
        self.x_values = x_train[argsort(x_train)]
        self.y_values = y_train[argsort(x_train)]
        self.pruning = p
        
        length = len(self.x_values)
        
        if length <= self.pruning:
            self.split_condition = False
        else:
            self.split_condition = True

        if self.split_condition == True:
            self.middle = (np.unique(self.x_values)[1:] + np.unique(self.x_values)[:-1]) / 2


    def split(self):
        if self.split_condition:
            decide = np.vectorize(self.decide)
            impurity = decide(self.middle)
            self.boundary = self.middle[impurity.argmin()]

            x_left = self.x_values[self.boundary > self.x_values]
            y_left = self.y_values[self.boundary > self.x_values]
            
            x_right = self.x_values[self.boundary <= self.x_values]
            y_right = self.y_values[self.boundary <= self.x_values]
            
            self.rightChild = BinaryTree(x_right, y_right, self.pruning)
            self.rightChild.split()
            
            self.leftChild = BinaryTree(x_left, y_left, self.pruning)
            self.leftChild.split()
            
    def decide(self, w0):
        x_left = self.x_values[self.x_values <= w0]
        x_right = self.x_values[self.x_values > w0]
        
        y_left = self.y_values[self.x_values <= w0]
        y_right = self.y_values[self.x_values > w0]

        Left = BinaryTree(x_left, y_left, self.pruning)
        Rigth = BinaryTree(x_right, y_right, self.pruning)

        iRight = np.sum(
            (Rigth.y_values - (np.sum(Rigth.y_values) / len(Rigth.x_values))) ** 2)
        iLeft = np.sum(
            (Left.y_values - (np.sum(Left.y_values) / len(Left.x_values))) ** 2)
       
        x_len = len(self.x_values)
        
        impurity = (iLeft + iRight) / x_len
        return impurity
    
    def rec(self, new_data_point):
        if not self.split_condition:
            y_len = len(self.y_values)
            return np.sum(self.y_values) /  y_len

        if new_data_point > self.boundary:
            return self.rightChild.rec(new_data_point)
        else:
            return self.leftChild.rec(new_data_point)



# In[5]:


regDecisionTree = BinaryTree(x_train, y_train, P)
regDecisionTree.split()

line = np.linspace(1, 6, 1001)
recursion_func = np.vectorize(regDecisionTree.rec)
y_hat_train = recursion_func(line)

plt.figure(figsize=(10, 4))
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.title("P = 25", fontweight="bold")
plt.plot(x_train, y_train, "b.", markersize=8, alpha=0.5, label="training")
plt.plot(x_test, y_test, "r.", markersize=8, alpha=0.5, label="test")
plt.legend(loc=2)
plt.plot(line, y_hat_train, 'k')
plt.show()

y_hat_test = recursion_func(x_test)


# In[6]:


y_hat_train = recursion_func(x_train)


# In[7]:


rmse = np.sqrt(np.sum((y_test - y_hat_test) ** 2) / N_test)
print("RMSE is", round(rmse, 7), "when P is", regDecisionTree.pruning)

rmse = np.sqrt(np.sum((y_train - y_hat_train) ** 2) / N_train)
print("RMSE is", round(rmse, 7), "when P is", regDecisionTree.pruning)


# In[8]:


p_arr = np.arange(5, 51, 5)
rmse_array = []

p_arr1 = np.arange(5, 51, 5)
rmse_array1 = []


# In[9]:


for p_val in p_arr:
    trainTree = BinaryTree(x_train, y_train, p_val)
    trainTree.split()
    recursion_f = np.vectorize(trainTree.rec)
    y_hat_iterative = recursion_f(x_test)
    rmse_iterative = np.sqrt(np.sum((y_test - y_hat_iterative) ** 2) / N_test) 
    rmse_array.append(rmse_iterative)

for p_val in p_arr1:
    testTree = BinaryTree(x_test, y_test, p_val)
    testTree.split()
    recursion_f = np.vectorize(testTree.rec)
    y_hat_iterative = recursion_f(x_test) 
    rmse_iterative = np.sqrt(np.sum((y_test - y_hat_iterative) ** 2) / N_test) 
    rmse_array1.append(rmse_iterative)   
    
plt.figure(figsize=(10, 10))
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.plot(p_arr, rmse_array, 'ro-',label="testing")
plt.plot(p_arr1, rmse_array1, 'bo-', label="training")
plt.yticks(np.arange(4, 8, .5))
plt.legend(loc=1)
plt.show()

