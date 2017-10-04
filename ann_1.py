#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# Reading Data

data = pd.read_csv("Churn_Modelling.csv")
X = data.iloc[:,3:13].values
y = data.iloc[:,13].values

# Categorical Data - One-Zero encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_1 = LabelEncoder()
X[:,1] = X_1.fit_transform(X[:,1])
X_2 = LabelEncoder()
X[:,2] = X_2.fit_transform(X[:,2])

OneX = OneHotEncoder(categorical_features = [1])
X = OneX.fit_transform(X).toarray()
X = X[:, 1:]

# Making training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.fit_transform(X_test)


# ANN 

import keras
from keras.models import Sequential
from keras.layers import Dense

# first input layer

classifier = Sequential()
classifier.add(Dense(activation = "relu", input_dim = 11, units = 6, kernel_initializer = "uniform"))

# Hidden Layer

classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))

# Output Layer

classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform" ))

# Compiling ANN
from keras import optimizers
sgd = optimizers.Adam(lr = 0.05)
classifier.compile(optimizer = sgd , loss ="binary_crossentropy", metrics = ["accuracy"])

# Fitting the data to ANN

classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

# Prediction
 
y_pred = classifier.predict(X_test)
y_pred_1 = (y_pred > 0.5)
 

# Homework Predicting Single input
    #Geography: France
    #Credit Score: 600
    #Gender: Male
    #Age: 40 years old
    #Tenure: 3 years
    #Balance: $60000
    #Number of Products: 2
    #Does this customer have a credit card ? Yes
    #Is this customer an Active Member: Yes
    #Estimated Salary: $50000


new_prediction = classifier.predict(SC.fit_transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
# Confusion matrix

from sklearn.metrics import confusion_matrix
C = confusion_matrix(Y_test, y_pred_1)


