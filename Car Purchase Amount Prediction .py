#!/usr/bin/env python
# coding: utf-8

# # #CAR PURCHASE AMOUNT PREDICTION

# # Importing Libraries

# In[1]:


import pandas as pd               #For Tables Manipulations
import numpy as np                #For Numerical Analysis
import matplotlib.pyplot as plt   #For Data Plotting and Visualization
import seaborn as sns             #For Data Plotting and Visualization


# # Importing Dataset

# In[2]:


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')


# In[3]:


car_df


# # Data Visualizing

# In[4]:


sns.pairplot(car_df)


# # Creating Testing and Training Dataset / Data Cleaning

# In[5]:


X = car_df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis = 1)


# In[6]:


X


# In[7]:


y = car_df['Car Purchase Amount']


# In[8]:


y


# In[9]:


X.shape


# In[10]:


y.shape


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()    #To Normalize Data b/w 0 and 1
X_scaled = scaler.fit_transform(X)


# In[12]:


X_scaled


# In[13]:


scaler.data_max_


# In[14]:


scaler.data_min_


# In[15]:


print(X_scaled[:,0])


# In[16]:


y = y.values.reshape(-1,1)


# In[17]:


y_scaled = scaler.fit_transform(y)


# In[18]:


y_scaled


# # Training the Model

# In[19]:


X_scaled.shape


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_scaled)   #default value for spliting is 25% for testing i.e test_size = 0.25 


# In[21]:


X_train.shape # 75% dataset used for training


# In[22]:


X_test.shape # 25% dataset used for testing


# In[23]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[24]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[25]:


epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)


# # Evaluating Model

# In[26]:


print(epochs_hist.history.keys())


# In[27]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[28]:


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth

X_Testing = np.array([[1, 50, 50000, 10985, 629312]])


# In[29]:


y_predict = model.predict(X_Testing)
y_predict.shape


# In[30]:


print('Expected Purchase Amount=', y_predict[:,0])

