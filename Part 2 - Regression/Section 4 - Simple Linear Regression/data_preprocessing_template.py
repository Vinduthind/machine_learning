# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/harwinder.singh/Desktop/Data2.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
dfx = pd.DataFrame(X)
dfy = pd.DataFrame(y)

# fill missing data 
# we use this when ever we need the complete sheet data 

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0 )
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# how to encode catagorial Data
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
dfx = pd.DataFrame(X)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
dfx = pd.DataFrame(X)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
train_x = pd.DataFrame(x_train)
test_x = pd.DataFrame(x_test)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""