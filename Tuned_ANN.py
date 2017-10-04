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

# Tuned ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()

    classifier.add(Dense(activation = "relu", input_dim = 11, units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
    ##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
    ##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
    ##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
    ##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))
    ##classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))

    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform" ))
    classifier.compile(optimizer = optimizer , loss ="binary_crossentropy", metrics = ["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameter = {'optimizer':['adam','rmsprop','sgd'],
             'batch_size':[5,20,25],
             'nb_epoch':[25,50]}
grid_s = GridSearchCV(estimator = classifier , param_grid = parameter, scoring = "accuracy", cv = 10)
grid_s = grid_s.fit(X_train,Y_train)
b_a = grid_s.best_params_
b_p = grid_s.best_score_



