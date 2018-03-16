
# coding: utf-8

# # Feature 

# In[5]:


import pandas as pd
import datetime
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
weekend = ['Saturday','Sunday']
def week_day_type(x):
    if x in weekend:
        return 'weekends'
    else:
        return 'weekdays'
def time_slot(x):
    if x in morning:
        return 'morning'
    elif x in afternoon:
        return 'afternoon'
    elif x in evening:
        return 'evening'
    else:
        return 'night'  
df=pd.read_csv("C:/Users/nitin/Documents/NEU/SEM 2/ADS/Assignment 2/Appliances-energy-prediction-data-master/energydata_complete.csv")
df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day']=df['date'].dt.day
df['day_of_week']=df['date'].dt.weekday_name
df['time_hr_24']=df['date'].dt.hour
df['time_min']=df['date'].dt.minute
df['week_day_type']=df['day_of_week'].map(week_day_type)
morning=range(6,12)
afternoon=range(12,17)
evening=range(17,22)  
df['time_slot']=df['time_hr_24'].map(time_slot)
df.drop(['date'],axis=1,inplace=True)
df=pd.get_dummies(df,prefix=['DOW','TS','WDT'],columns=['day_of_week','time_slot','week_day_type'])
print(df.shape)


# Let's check what the dataset contains.

# In[141]:


df


# The dataset is huge. Before performing any mathematical calculations, we have to know how much data does it contains.

# In[142]:


df.shape


# So the data contains 29 columns and 19,735 rows.

# We also have to find out how many null values are present in our dataset.

# In[143]:


df.isnull().sum()


# As we can see, we don't have any null values present in any row of our dataset. We also have to understand the nature of these values before performing any mathematical calculations. 

# In[145]:


df.dtypes


# We can see that all importamt data are in 'float' format. We can also observe that few columns have boolean data. This is the result of 'One-hot encoding'. 

# Spliting data and normalization

# In[6]:


df_train,df_test = train_test_split(df,train_size=0.7,random_state=42)
x_train=df_train.iloc[:,1:]
y_train=df_train['Appliances']
scaler.fit(x_train)
x_train_sc=scaler.transform(x_train)
x_test=df_test.iloc[:,1:]
y_test=df_test['Appliances']
x_test_sc=scaler.transform(x_test)


# Linear Regression Model

# In[10]:


lm=linear_model.LinearRegression()
mod=lm.fit(x_train_sc,y_train)
print(mod.coef_)
print(x_train.columns)


# Random Forest Model

# In[71]:


rf=RandomForestRegressor()
rf.fit(x_train_sc, y_train)
feature_list = list(x_train.columns)
importances = list(rf.feature_importances_)
feature_importances = [(x_train, round(importance, 2)) for x_train, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Let's try to understand the importance of these features through a graph

# In[140]:



a=d['Feature']
plt.xlabel("Value" , fontsize=20)
plt.ylabel("Feature" , fontsize=20)
plt.legend()
plt.title("Importance of Features" , fontsize=25)
plt.hlines(y=a, xmin=0, xmax=d['Value'], color='skyblue')
plt.plot(d['Value'], a, "o")
plt.rcParams['figure.figsize'] = (15 , 3)
plt.show()

