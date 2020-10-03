#!/usr/bin/env python
# coding: utf-8

# ## Name - Puspita Saha

# # Decision Tree Classification

# ## Importing the libraries

# In[45]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[46]:


dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[48]:


print(X_train)


# In[49]:


print(y_train)


# In[50]:


print(X_test)


# In[51]:


print(y_test)


# ## Feature Scaling

# In[52]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[53]:


print(X_train)


# In[54]:


print(X_test)


# ## Training the Decision Tree Classification model on the Training set

# In[55]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[56]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix

# In[57]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[58]:


import sklearn.datasets as datasets
iris=datasets.load_iris()

df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(5))

y=iris.target
print(y)


# In[59]:


# Install required libraries
get_ipython().system('pip install pydotplus')
get_ipython().system('apt-get install graphviz -y')


# Visualizing the decision tree graph

# In[60]:


# Import necessary libraries for graph viz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()
export_graphviz(classifier,out_file=dot_data, feature_names=iris.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

