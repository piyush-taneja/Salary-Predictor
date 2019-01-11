# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 01:52:04 2018

@author: piyush taneja
"""
#Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Make sure X is always matrix that why we used 1:2 and not just 1
y = dataset.iloc[:, 2].values    # y is to be made as a vector

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
# Here we do not divide the data into training and test set as data only 10 observations. So divinding would not be a great choice.

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
    
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)  # Here it automatically takes the constant term column i.e b0 which contains all 1's but in multiple we need to add it using append function
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly , y)

#Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X,lin_reg.predict(X) , color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape((len(X_grid) , 1))  # Reshaping the vector of X_Grid into matrix
plt.scatter(X, y, color='red')
plt.plot(X_grid , lin_reg_2.predict(poly_reg.fit_transform(X_grid)) , color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)   # This tells the salary at the level 6.5

#Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))