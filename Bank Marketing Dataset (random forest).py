#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#creation of base dataset
data_term_dep= pd.read_csv("/Users/Admin/Downloads/bank_customer_survey.csv")


# In[3]:


data_term_dep.head()


# In[4]:


data_term_dep.shape


# In[5]:


#null value treatment

def null_values(base_dataset):
    print(base_dataset.isna().sum())
    ##null value percentage
    null_value_table=(base_dataset.isna().sum()/base_dataset.shape[0])*100
    ##null value percentage beyond threshold drop, else treat the columns
    retained_columns=null_value_table[null_value_table<30].index
    #if any variable as null value greater than input(like 30% of the data) value than those variable
    #are considered as drop
    drop_columns= null_value_table[null_value_table>30].index
    base_dataset.drop(drop_columns,axis=1,inplace=True)
    len(base_dataset.isna().sum().index)
    cont=base_dataset.describe().columns
    cat=[i for i in base_dataset.columns if i not in base_dataset.describe().columns]
    for i in cat:
        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)
    for i in cont:
        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)
    print(base_dataset.isna().sum())
    return base_dataset,cat,cont
                        


# In[6]:


data_term_dep1,cat,cont= null_values(data_term_dep)


# In[7]:


#outlier treatment

def outliers_transform(base_dataset):
    for i in base_dataset.var().sort_values(ascending=False).index[0:10]:
        x=np.array(base_dataset[i])
        qr1=np.quantile(x,0.25)
        qr3=np.quantile(x,0.75)
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        y=[]
        for p in x:
            if p <ltv or p>utv:
                y.append(np.median(x))
            else:
                y.append(p)
        base_dataset[i]=y       


# In[8]:


outliers_transform(data_term_dep1)


# In[9]:


#display the columns after outlier treatment
data_term_dep1.columns


# In[10]:


#dummy variable declaration

dummy_columns=[]
for i in data_term_dep1.columns:
    if(data_term_dep1[i].nunique()>=3) & (data_term_dep1[i].nunique()<=5):
        dummy_columns.append(i)
        
#nunique() function returns no of unique elements in the object.it returns a scalar value which is
#the count of all the unique values in the Index


# In[11]:


dummy_columns


# In[12]:


#Dummy Variable
dummies_tables=pd.get_dummies(data_term_dep1[dummy_columns])


# In[13]:


for i in dummies_tables.columns:
    data_term_dep1[i]=dummies_tables[i]


# In[14]:


#displaying columns after dummy variable creation
data_term_dep1.columns


# In[15]:


#drop the existing columns after the creation of dummy variable for those
data_term_dep1=data_term_dep1.drop(dummy_columns,axis=1)


# In[16]:


data_term_dep1.columns


# In[17]:


#Label Encoder
#Label Encoding refers to converting the labels into numeric form so as to convert it into the
#machine-readable form. Machine learning algorithms can then decide in a better way on how those
#labels must be operated. It is an important pre-processing step for the structured dataset in
#supervised learning

from sklearn.preprocessing import LabelEncoder
def label_encoders(data,cat):
    le=LabelEncoder()
    for i in cat:
        le.fit(data[i])
        x=le.transform(data[i])
        data[i]=x
    return data

#Transform function returns a self-produced dataframe with transformed values after applying the
#function specified in its parameter


# In[18]:


data_new=data_term_dep1
cat=data_term_dep1.describe(include='object').columns


# In[19]:


label_encoders(data_new,cat).head()


# In[20]:


data_new.columns


# In[21]:


#univariate analysis(EDA)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
for i in data_new.var().index:
    sns.distplot(data_new[i],kde=False)
    plt.show()
    
#Kernel Density Estimation (KDE) is a way to estimate the probability density function of a continuous
#random variable. By default, seaborn plots both kernel density estimation and histogram, kde=False
#means you want to hide it and only display the histogram
#A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in
#a dataset, analagous to a histogram. ... But it has the potential to introduce distortions if the
#underlying distribution is bounded or not smooth    


# In[22]:


#bivariate analysis(EDA)

plt.figure(figsize=(20,10))
sns.heatmap(data_new.corr())


# In[23]:


#model building

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


# In[24]:


y=data_new['y']
x=data_new.drop('y',axis=1)


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=43)


# In[26]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# In[27]:


models=[DecisionTreeClassifier(),RandomForestClassifier(),BaggingClassifier()]


# In[29]:


from sklearn.metrics import confusion_matrix,accuracy_score
final_accuracy_scores=[]
for i in models:
    dt=i
    dt.fit(x_train,y_train)
    dt.predict(x_test)
    dt.predict(x_train)
    print(confusion_matrix(y_test,dt.predict(x_test)))
    print(accuracy_score(y_test,dt.predict(x_test)))
    print(confusion_matrix(y_train,dt.predict(x_train)))
    print(accuracy_score(y_train,dt.predict(x_train)))
    print(i)
    final_accuracy_scores.append([i,confusion_matrix(y_test,dt.predict(x_test)),
    accuracy_score(y_test,dt.predict(x_test)),confusion_matrix(y_train,dt.predict(x_train)),
    accuracy_score(y_train,dt.predict(x_train))])
    from sklearn.model_selection import cross_val_score
    print(cross_val_score(i,x_train,y_train,cv=10))


# In[30]:


final_accuracy_scores1=pd.DataFrame(final_accuracy_scores)


# In[31]:


final_accuracy_scores1


# In[32]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,dt.predict(x_test))


# In[ ]:




