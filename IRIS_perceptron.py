#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:27:53 2019

@author: adithya, rahul
"""

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def model1():
    """
    The outline of model. It has three fully-connected (Dense) layers.
    
    Returns:
        model -- The model which has been designed
    """
    model = Sequential()
    model.add(Dense(20, input_dim=4, activation='relu'))
    model.add(Dense(10, input_dim=20, activation='relu'))
    model.add(Dense(3, input_dim=10, activation='softmax'))
    
    return model

# Reading the data using pandas. We get the output as a dataframe (df) 
# 'header=0' ensures that the first row is considered as header
# We don't use any pre-processign because data is already clean
df = pandas.read_csv('iris-species/iris.csv', header = 0)

X = df.values[:,1:5].astype('float') # columns 2 through 5 are having data about flower
Y = df.values[:,5]# column 6 contains the type of flower

# In the next four lines, we are converting the data in Y to its one-hot representation
# i.e. we are creating a separate column for each class (species) and 
# assigning 1 if it belongs to that class and 0 if it does not
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)

#We are splitting the data for training and testing using test_train_split from scikit-learn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

model = model1()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# validation split further splits the training data to validation and training
# validation data helps check if the model is overfitting
model.fit(X_train, Y_train, epochs = 200, validation_split = 0.1)

# model.evaluate(...) returns a list with the loss and accuracy in that order
print(model.evaluate(X_test, Y_test))

