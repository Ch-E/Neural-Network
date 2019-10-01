# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:51:15 2019

@author: Charl
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')
train.head()
train.describe()
train.info()
train.isnull().values.any()



train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train["skill"] = train["headshotKills"]+train["roadKills"]
train.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(train.shape)
train.head()

#test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
#test["skill"] = test["headshotKills"]+test["roadKills"]
#test.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
#print(test.shape)
#test.head()

predictors = ["kills",
              "killStreaks",
              "killPlace",
              "maxPlace",
              "numGroups",
              "distance",
              "boosts",
              "weaponsAcquired",
              "DBNOs",
               ]

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



# PAIRPLOT
#sns.pairplot(data.drop(['date','condition','yr_built','zipcode','waterfront','is_basement'],axis=1))
#data.groupby('waterfront').mean()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

norm_X_train = scaler.fit_transform(X_train)

norm_X_test = scaler.fit_transform(X_test)

norm_X_train = pd.DataFrame(data=norm_X_train,columns=X_train.columns)

norm_X_test = pd.DataFrame(data=norm_X_test,columns=X_test.columns)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def build_model():
    model = keras.Sequential([
        layers.Dense(60, activation=tf.nn.relu, input_shape=[len(norm_X_train.keys())]),
        layers.Dropout(0.5),
        layers.Dense(60, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)),
        layers.Dropout(0.5),
        layers.Dense(60, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)),
        layers.Dense(1)
          ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()

model.summary()

EPOCHS = 1500

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    
    plt.legend()
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    
    plt.legend()
    plt.show()

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(norm_X_train, y_train, epochs=EPOCHS,batch_size=100,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop])

plot_history(history)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

loss, mae, mse = model.evaluate(norm_X_train, y_train, verbose=0)

print("Training set Mean Abs Error: {:5.2f}".format(mae))

train['winPlacePerc'].mean()

test_predictions = model.predict(norm_X_test).flatten()
plt.figure(figsize=(10,6))
plt.scatter(y_test, test_predictions)
plt.xlabel('True prices')
plt.ylabel('Predicted prices')
plt.xlim(0,2000000)
plt.ylim(0,2000000)

loss, mae, mse = model.evaluate(norm_X_test, y_test, verbose=0)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))

#using linear regression - 
# from sklearn.linear_model import LinearRegression
# lm = LinearRegression()
# lm.fit(norm_X_train,y_train)
# predictions = lm.predict(norm_X_test)
# plt.figure(figsize=(10,6))
# ax = plt.scatter(y_test,predictions)
# plt.xlim(0,2000000)

# from sklearn import metrics
# metrics.mean_absolute_error(y_test,predictions)





































