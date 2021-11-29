#!/usr/bin/env python
# coding: utf-8

# # Neha Bheemisetty

# # LGM VIP ADVANCE LEVEL TASK - 1 (DATA SCIENCE)

# # Develop A Neural Network That Can Read Handwriting!!

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[4]:


train


# In[5]:


test


# In[6]:


print(train.shape)
print(test.shape)


# In[7]:


train.head(5)


# In[8]:


test.head(5)


# In[9]:


train.describe()


# In[10]:


test.describe()


# In[11]:


train.info()


# In[12]:


train.columns


# In[14]:


test.value_counts()


# In[15]:


train.dtypes


# In[16]:


X=train.drop(["label"],axis=1).values
Y=train["label"].values


# In[17]:


train.corr()


# In[18]:


test.corr()


# # Exploratory Data Analysis

# In[19]:


plt.figure(figsize=(16,8))
sns.histplot(Y)


# In[20]:


plt.figure(figsize=(16,8))
sns.distplot(Y,color='g')


# In[22]:


plt.figure(figsize=(16,8))
sns.kdeplot(Y,shade=True)


# In[23]:


plt.figure(figsize=(16,8))
sns.violinplot(Y,color='g')


# In[24]:


plt.figure(figsize=(16,8))
sns.countplot(Y)


# In[26]:


plt.imshow(X[10].reshape([28,28]))


# In[27]:


plt.imshow(X[1455].reshape([28,28]))


# In[28]:


X= X.reshape([42000,28,28,1])
Y=Y.reshape([42000,1])


# In[30]:


from keras.utils.np_utils import to_categorical
Y=to_categorical(Y,num_classes=10)


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=10,test_size=0.1)


# In[32]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax')
])


# In[33]:


model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['binary_accuracy'],
)


# In[34]:


history=model.fit(x_train,y_train,batch_size=256,
                 epochs=20
)


# In[35]:


model.summary()


# In[36]:


from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)


# In[37]:


y_pred=model.predict(x_test)
y_pred


# In[38]:


print(y_pred>0.5)


# In[39]:


history_df=pd.DataFrame(history.history)
history_df['loss'].plot()


# In[40]:


model.evaluate(x_test,y_test)


# Conclusion:
# 
# So, we got a good accuracy of 99.8% keeping epochs as 20 and batch size as 256
# 
