"""
Created on Monday, Sep 4, 2017

@author: Mohamed Ameen

@title: Titanic - Machine Learning from Disaster

@description:
    Predicting Titanic survival probability using K-Nearest Neighbors (K-NN)
    Accuracy = 74.88%
"""

#Importing the libraries
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.nan)

#Importing the dataset
dataset = pd.read_csv("Data/train.csv")
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:, 1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix to calculate the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
Accuracy = ((cm[0, 0] + cm[1, 1]) / len(X_test)) * 100
