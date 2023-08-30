#!/usr/bin/env python
# coding: utf-8

# # Lets Grow More(LGMVIP)
# ## Task 2: Stock Market Prediction and forecasting using stacked LSTM
# ### Author: Omkar Sanjay Chavan

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:/Omkar/raw.githubusercontent.com_mwitiderrick_stockprice_master_NSE-TATAGLOBAL.csv")


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.columns


# In[9]:


df1=df.reset_index()['Close']
df1


# In[10]:


df.head()


# In[11]:


df['Date']=pd.to_datetime(df["Date"])
df.info()


# In[12]:


df1=df.reset_index()["Close"]


# In[13]:


df1


# In[14]:



plt.plot(df1)


# ### Scaling

# In[15]:



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[16]:


df1


# ### Training of data

# In[17]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[18]:


training_size,test_size


# In[19]:


import numpy
def create_dataset(dataset,time_step=1):
    dataX, dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a= dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i +time_step,0])
    return numpy.array(dataX), numpy.array(dataY)


# In[20]:


time_step=100
X_train,y_train=create_dataset(train_data, time_step)
X_test,y_test = create_dataset(test_data, time_step)


# In[21]:


print(X_train)


# In[22]:


print(X_train.shape), print(y_train.shape)


# In[23]:


print(X_test.shape), print(y_test.shape)


# In[24]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[26]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1)) 
model.compile(loss='mean_squared_error', optimizer='adam')


# In[27]:


model.summary()


# ### Training the model

# In[28]:


model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=100,batch_size=64, verbose=1)


# In[29]:


import tensorflow as tf


# In[30]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[31]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[32]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_test,test_predict))


# In[33]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[34]:


look_back=100
trainPredictPlot=numpy.empty_like(df1)
trainPredictPlot[:, :]= np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :]= train_predict

testPredictPlot=numpy.empty_like(df1)
testPredictPlot[:, :]= np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]= test_predict

plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

