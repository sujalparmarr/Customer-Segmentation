#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[3]:


#loading data from csv file
data=pd.read_csv(r"C:\Users\sujal\Desktop\Data Science\Mall_Customers.csv")
data.head()


# In[5]:


#total number of rows and columns
data.shape


# In[8]:


data.info()


# In[9]:


#checking for missing values
data.isnull().sum()


# In[10]:


# choosing columns Annual income and Spending Score
X=data.iloc[:,[3,4]].values


# In[12]:


print(X)


# In[14]:


# Choosing the numbers of clusters
#finding wcss value of different number of clusters
wcss=[]
for i in range (1,11):
    kmeans= KMeans(n_clusters=i, init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[18]:


sns.set()
plt.plot(range(1,11
              ),wcss)
plt.title("The Elbow Point Graph")
plt.xlabel("Number of clusters")
plt.ylabel("Wcss")
plt.show()

Optimum Number of Clusters = 5

# In[21]:


#Training KMeans clustering model
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
#return a label for each data point based on their cluster
Y=kmeans.fit_predict(X)
print(Y)


# In[24]:


#visualization all the clusters
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plotting  the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:




