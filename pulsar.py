# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:14:30 2020

@author: Sejal Vyas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:,:8].values
y = dataset.iloc[:,8].values

#Missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)


#Splitting data into Test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

tp = cm[0][0]
fn = cm[0][1]
fp = cm[1][0]
tn = cm[1][1]

accuracy = (tp + tn)/(tp + fp + tn + fn)
precision = (tp)/(tp+fp)
recall = tp/(tp+fn)
f1 = 2* precision * recall / (precision + recall)
