# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 16:31:14 2017

@author: Anu
"""

#Data preprocessing

# Importing the libraries
import numpy 
import matplotlib.pyplot
import pandas

# Importing the dataset
dataset= pandas.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:,3].values

#missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:, 1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])

#encoding categorical variable
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_x=LabelEncoder()
x[: ,0]=labelencoder_x.fit_transform(x[: ,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
X_train=Sc_X.fit_transform(X_train)
X_test=Sc_X.transform(X_test)

