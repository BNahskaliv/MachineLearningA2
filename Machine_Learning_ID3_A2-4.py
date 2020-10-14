#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Importing prerequisite modules
import math
import pandas as pnd
import numpy  as nmp
from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split
from pprint                  import pprint


# In[12]:


# Importing the Iris dataset
Data = load_iris()


# In[13]:


# Column Names
for i in Data.feature_names:
    print (i)


# In[14]:


# Getting Dimension & Size of dataset
Data.data.shape


# In[15]:


x = Data.data
y = Data.target

# Creating Test & Train data with 80% and 20% segregation
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[16]:


# Estimating the entropy for the split
def Entropy(A,B):
    # Total rows of records
    N=A+B
    # When the split has got only one class
    if A==0 or B==0:
        return 0
    else:
        return -(A/N)*math.log(A/N,2) - (B/N)*math.log(B/N,2)


# In[17]:


# Estimating the  entropy for branch
def Grp_Entropy(node): 
    
    
    Grp_Ent = 0                         # Group Entropy value
    
    N = len(node)                       # Total number rows in a group
    
    Class = set(node)                   # Checking for Unique Classes
    
    for i in Class:                     # Weighted Average Entropy for each respective class
        C = sum(node==i)
    
        E = C/N * Entropy(sum(node==i),sum(node!=i)) 
        Grp_Ent += E
    return Grp_Ent, N


# In[18]:


# The combined entropy of two big circles.
def Overall_Entropy(Predict,Real):
    
    if len(Predict) != len(Real):                      # Lenght must be same
        return None
    
    N = len(Real)
    
    
    Total_Left, n_true = Grp_Entropy(Real[Predict])    # Left Split Entropy Calculation
    
    Total_Right, n_false = Grp_Entropy(Real[~Predict]) # Right Split Entropy Calculation
    
    
    Overall_Entropy = n_true/N * Total_Left + n_false/N * Total_Right # Calculating The Entropy Overall
    
    return Overall_Entropy


# In[19]:


class DTClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
    
    def fit(self, x, y, node={}, depth=0):
        if node is None: 
            return None
        elif len(y) == 0:
            return None
        elif self.samecheck(y):
            return {'val':y[0]}
        elif depth >= self.max_depth:
            return None
        else: 
            col, cutoff, entropy = self.best_split_search(x, y)    # find one split given an information gain 
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            node = {'col': Data.feature_names[col], 'index_col':col,
                        'cutoff':cutoff,
                       'val': nmp.round(nmp.mean(y))}
            node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth+1)
            node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth+1)
            self.depth += 1 
            self.trees = node
            return node
    
    def best_split_search(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.bestsplit(c, y)
            if entropy == 0:                         # look for the first optimum cutoff & pause the iteration
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy
    
    def bestsplit(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = Overall_Entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff
    
    def samecheck(self, items):
        return all(x == items[0] for x in items)
                                           
    def predict(self, x):
        tree = self.trees
        results = nmp.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self.getpred(c)
        return results
    
    def getpred(self, row):
        cur_layer = self.trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')


# In[20]:


clf = DTClassifier(max_depth=7)
m = clf.fit(X_train, y_train)

pprint(m)


# In[21]:


# Predicting the fields
A=clf.predict(X_test)


# In[23]:


# Accuracy Calculation Of Algorithm
count=0
for i in range(len(A)):
    if A[i]==y_test[i]:
        count+=1
Acc=(count*100)/len(A)
print (Acc)


# In[ ]:





# In[ ]:




