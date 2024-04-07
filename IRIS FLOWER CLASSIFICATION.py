#!/usr/bin/env python
# coding: utf-8

# # IRES FLOWER CLASSIFICATION

# In[1]:


#importing basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score


# In[24]:


data_set=pd.read_csv(r"C:\Users\HP\OneDrive\Documents\oasis infobytes\Iris.csv")
data_set.head()


# In[3]:


data_set.tail() 


# In[4]:


data_set["Species"].unique()


# In[ ]:





# In[ ]:





# # checking missing value

# In[5]:


data_set.isnull().sum()


# In[6]:


data_set.info()


# In[7]:


type(data_set) #checking the type of dataset


# In[8]:


print(data_set.shape) #In this csv file we have 300 rows and 5 columns .


# In[9]:


data_set.describe()


# In[ ]:





# In[ ]:





# # Data visualizataion

# In[10]:


sns.pairplot(data_set,hue="Species")
plt.show()


# In[11]:


sns.lmplot(x='PetalLengthCm',y='PetalWidthCm',data=data_set)


# In[12]:


sns.heatmap(data_set.corr(),annot=True)


# In[13]:


plt.title('Ires flower sepalWidthcm and sepalWidthCm') 
sns.lineplot(x='SepalLengthCm', y='SepalWidthCm', data=data_set,color="red")


# In[14]:


plt.title('Ires flower PetalWidthcm and PetalWidthCm') 
sns.lineplot(x='PetalLengthCm', y='PetalWidthCm', data=data_set,color="green")


# In[ ]:





# In[ ]:





# # Modifiying the data

# In[15]:


X=data_set.drop("Species",axis=1)
X


# In[16]:


Y=data_set["Species"]
Y


# In[ ]:





# In[ ]:





# # Training the model

# In[17]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)


# In[18]:


knn=KNeighborsClassifier(n_neighbors=3) #KNeihboursClassification Algorithm.
knn.fit(x_train, y_train)


# In[19]:


y_predict=knn.predict(x_test)


# In[20]:


print("Accuracy:" ,accuracy_score(y_test,y_predict))


# In[ ]:





# In[ ]:





# # Finally created model prediction (or) Final Result.

# In[21]:


import pandas as pd
new_dataset=pd.DataFrame({"SepalLengthCm":2.3,"SepalWidthCm":3.0,"PetalLengthCm":4.2,"PetalWidthCm":7.2}, index=[0])
prediction=knn.predict(new_dataset)
prediction[0]


# In[ ]:





# In[22]:


#Thanking you...


#                                                                                                        - Thiruvalluvan G
