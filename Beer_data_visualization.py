
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import numpy as np
fs = 16


# In[2]:


df =  pd.read_excel('New_beer_dataset.xlsx')
df1 =  pd.read_excel('cases_beer.xlsx')


# In[3]:


df.head(10)


# In[4]:


df1.head(10)


# In[5]:


plt.figure(figsize=(8,5))
plt.plot(df['Week'], df['PRICE 12PK'], '-')
plt.plot(df['Week'], df['PRICE 18PK'], '-')
plt.plot(df['Week'], df['PRICE 30PK'], '-')
plt.legend(['PRICE 12PK', 'PRICE 18PK', 'PRICE 30PK'])
plt.ylabel("Price", fontsize = fs)
plt.xlabel("Week", fontsize = fs)
plt.show()


# In[6]:


plt.figure(figsize=(8,5))
plt.plot(df1['Week'], df1['CASES 12PK'], '-')
plt.plot(df1['Week'], df1['CASES 18PK'], '-')
plt.plot(df1['Week'], df1['CASES 30PK'], '-')
plt.legend(['CASES 12PK', 'CASES 18PK', 'CASES 30PK'])
plt.ylabel("Volume", fontsize = fs)
plt.xlabel("Week", fontsize = fs)
plt.show()


# In[7]:


#Combine the two DataFrame
combined_df = pd.merge(df, df1, on="Week")


# In[8]:


#Correlation Representation`
corrmat = combined_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# In[9]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
reg = linear_model.LinearRegression()

#Splitting the entire dataset into testing and training set
X_train, X_test, y_train, y_test = train_test_split(combined_df['PRICE 12PK'], combined_df['CASES 12PK'], test_size=0.20)

#Training the Regression Model
reg.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

#Predicting sales volume for the test set
y_predict = reg.predict(X_test.reshape(-1,1))
plt.ylabel("Volume", fontsize = fs)
plt.xlabel("Price", fontsize = fs)
plt.plot(X_train, y_train, 'o')
plt.plot(X_test, y_predict,'-')
plt.show()
rms = sqrt(mean_squared_error(y_test.reshape(-1,1) , y_predict))
print("Root mean square error ",rms)


# In[10]:


#Splitting the entire dataset into testing and training set
X_train, X_test, y_train, y_test = train_test_split(combined_df['PRICE 18PK'], combined_df['CASES 18PK'], test_size=0.20)

#Training the Regression Model
reg.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

#Predicting sales volume for the test set
y_predict = reg.predict(X_test.reshape(-1,1))
plt.ylabel("Volume", fontsize = fs)
plt.xlabel("Price", fontsize = fs)
plt.plot(X_train, y_train, 'o')
plt.plot(X_test, y_predict,'-')
plt.show()

rms = sqrt(mean_squared_error(y_test.reshape(-1,1) , y_predict))
print("Root mean square error ",rms)


# In[13]:


#Splitting the entire dataset into testing and training set
X_train, X_test, y_train, y_test = train_test_split(combined_df['PRICE 30PK'], combined_df['CASES 30PK'], test_size=0.20)

#Training the Regression Model
reg.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

#Predicting sales volume for the test set
y_predict = reg.predict(X_test.reshape(-1,1))
plt.ylabel("Volume", fontsize = fs)
plt.xlabel("Price", fontsize = fs)
plt.plot(X_train, y_train, 'o')
plt.plot(X_test, y_predict,'-')
plt.show()

rms = sqrt(mean_squared_error(y_test.reshape(-1,1) , y_predict))

print("Actual value: ",y_test)
print("\n")
print("Predicted value: ",y_predict)
print("Root mean square error ",rms)

