"""
Created on Tue Feb  6 10:46:32 2018
Last Updated on Thurs Feb 8 11:52:28 2018
@author: Victor Hernandez
"""
from utils import read_data
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing as prepros
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn import svm
from sklearn.neural_network import MLPRegressor as MLPR
import matplotlib.pyplot as plt

Features, Labels = read_data("energy.txt")
X_train, X_test, y_train, y_test = train_test_split(Features, Labels, test_size = .20, random_state=37)

#Standardization of data 
X_train_scaled = prepros.scale(X_train)
X_test_scaled = prepros.scale(X_test)

"""
####################K-Nearest Neighbors####################
"""
#No changes
KNNregressor = KNR().fit(X_train, y_train)
KNN_y_pred = KNNregressor.predict(X_test)
print("Mean squared error for K-Nearest Neighbors: {:.3f}.".format(mean_squared_error(KNN_y_pred,y_test)))

#number of neigbors changed to 7
KNNregressor = KNR(n_neighbors = 7).fit(X_train, y_train)
KNN_y_pred = KNNregressor.predict(X_test)
print("\tNumber of Neigbors changed to 7: {:.3f}.".format(mean_squared_error(KNN_y_pred,y_test)))
  

#wieght changed to distance
KNNregressor = KNR(weights = 'distance').fit(X_train, y_train)
KNN_y_pred = KNNregressor.predict(X_test)
print("\tWieght changed to distance: {:.3f}.".format(mean_squared_error(KNN_y_pred,y_test)))

#Manhattan distance used
KNNregressor = KNR(p = 1).fit(X_train, y_train)
KNN_y_pred = KNNregressor.predict(X_test)
print("\tManhattan distance used: {:.3f}.".format(mean_squared_error(KNN_y_pred,y_test)))

#Data Standadized 
KNNregressor = KNR().fit(X_train_scaled, y_train)
KNN_y_pred = KNNregressor.predict(X_test_scaled)
print("\tData standardized: {:.3f}.".format(mean_squared_error(KNN_y_pred,y_test)))

#All changes applied
KNNregressor = KNR(n_neighbors = 7,  weights='distance', p = 1).fit(X_train_scaled, y_train)
KNN_y_pred = KNNregressor.predict(X_test_scaled)
print("\tAll changes applied: {:.3f}.".format(mean_squared_error(KNN_y_pred,y_test)))
plt.scatter(KNN_y_pred, y_test)  

"""
####################Linear Regression####################
"""
lr = LR().fit(X_train, y_train)
LinReg_y_pred = lr.predict(X_test)
print("\nMean squared error for Linear Regression: {:.3f}.".format(mean_squared_error(LinReg_y_pred, y_test)))
plt.scatter(LinReg_y_pred, y_test, c = 'green')  

"""
####################Ridge Regression####################
"""
ridge = Ridge().fit(X_train, y_train)
Ridge_y_pred = ridge.predict(X_test)
print("\nMean squared error for Ridge Regression: {:.3f}.".format(mean_squared_error(Ridge_y_pred, y_test)))
plt.scatter(Ridge_y_pred, y_test, c = 'red')  

"""
####################Regression Tree####################
"""
#No changes
R_tree = DTR(random_state = 37).fit(X_train, y_train)
RT_y_pred = R_tree.predict(X_test)
print("\nMean squared error for Regression Tree: {:.3f}.".format(mean_squared_error(RT_y_pred, y_test)))

#Depth set to 11
R_tree = DTR(max_depth = 11, random_state = 37).fit(X_train, y_train)
RT_y_pred = R_tree.predict(X_test)
print("\tDepth set to 11: {:.3f}.".format(mean_squared_error(RT_y_pred, y_test)))

#Min samples split set to 16
R_tree = DTR(min_samples_split = 16, random_state = 37).fit(X_train, y_train)
RT_y_pred = R_tree.predict(X_test)
print("\tMin samples split set to 16: {:.3f}.".format(mean_squared_error(RT_y_pred, y_test)))

#Min samples leaf set to 2
R_tree = DTR(min_samples_leaf = 2, random_state = 37).fit(X_train, y_train)
RT_y_pred = R_tree.predict(X_test)
print("\tMin samples leaf set to 2: {:.3f}.".format(mean_squared_error(RT_y_pred, y_test)))

#Data Standardized
R_tree = DTR(max_depth = 11, random_state = 37).fit(X_train_scaled, y_train)
RT_y_pred = R_tree.predict(X_test_scaled)
print("\tData standardized: {:.3f}.".format(mean_squared_error(RT_y_pred, y_test)))

#All changes applied
R_tree = DTR(max_depth = 11, min_samples_split = 16, min_samples_leaf = 2, random_state = 37).fit(X_train_scaled, y_train)
RT_y_pred = R_tree.predict(X_test_scaled)
print("\tAll changes applied: {:.3f}.".format(mean_squared_error(RT_y_pred, y_test)))
plt.scatter(RT_y_pred, y_test, c ='orange')  

"""
####################Random Forests####################
"""
#No changes 
forest = RFR(n_jobs = -1, random_state = 37).fit(X_train, y_train)
F_y_pred = forest.predict(X_test)
print("\nMean squared error for Random Forests: {:.3f}.".format(mean_squared_error(F_y_pred, y_test)))

#Num of estimators 
forest = RFR(n_estimators = 100, n_jobs = -1, random_state = 37).fit(X_train, y_train)
F_y_pred = forest.predict(X_test)
print("\tNum of estimators set to 100: {:.3f}.".format(mean_squared_error(F_y_pred, y_test)))

#Min samples leaf set to 2
forest = RFR(min_samples_leaf = 2, n_jobs = -1, random_state = 37).fit(X_train, y_train)
F_y_pred = forest.predict(X_test)
print("\tMin samples leaf set to 2: {:.3f}.".format(mean_squared_error(F_y_pred, y_test)))

#Max leaf nodes set to 1000
forest = RFR(max_leaf_nodes = 1000, n_jobs = -1, random_state = 37).fit(X_train, y_train)
F_y_pred = forest.predict(X_test)
print("\tMax leaf nodes set to 1000: {:.3f}.".format(mean_squared_error(F_y_pred, y_test)))

#All changes applied
forest = RFR(n_estimators = 100, min_samples_leaf = 2, max_leaf_nodes = 1000, n_jobs = -1, random_state = 37).fit(X_train, y_train)
F_y_pred = forest.predict(X_test)
print("\tAll changes applied: {:.3f}.".format(mean_squared_error(F_y_pred, y_test)))
plt.scatter(F_y_pred, y_test, c = 'purple')  

"""
####################Support Vector Regresssion####################
"""
#No changes
SVR = svm.SVR().fit(X_train, y_train)
SVR_y_pred = SVR.predict(X_test)
print("\nMean squared error for Support Vector Regression: {:.3f}.".format(mean_squared_error(SVR_y_pred, y_test)))

#C set to 10
SVR = svm.SVR(C = 10).fit(X_train, y_train)
SVR_y_pred = SVR.predict(X_test)
print("\tC set to 10: {:.3f}.".format(mean_squared_error(SVR_y_pred, y_test)))

#Epsilon set to .01
SVR = svm.SVR(epsilon = .01).fit(X_train, y_train)
SVR_y_pred = SVR.predict(X_test)
print("\tEpsilon set to .01: {:.3f}.".format(mean_squared_error(SVR_y_pred, y_test)))

#Gamma set to .03
SVR = svm.SVR(gamma = .03).fit(X_train, y_train)
SVR_y_pred = SVR.predict(X_test)
print("\tGamma set to .03: {:.3f}.".format(mean_squared_error(SVR_y_pred, y_test)))

#data standardized
SVR = svm.SVR().fit(X_train_scaled, y_train)
SVR_y_pred = SVR.predict(X_test_scaled)
print("\tData standardized: {:.3f}.".format(mean_squared_error(SVR_y_pred, y_test)))

#All changes applied
SVR = svm.SVR(C = 10, epsilon = .01, gamma = 1).fit(X_train_scaled, y_train)
SVR_y_pred = SVR.predict(X_test_scaled)
print("\tAll changes applied: {:.3f}.".format(mean_squared_error(SVR_y_pred, y_test)))
plt.scatter(SVR_y_pred, y_test, c = 'violet')  

"""
####################Multilayer Perceptron####################
"""
#No changes 
MLP = MLPR(random_state = 37).fit(X_train, y_train)
MLP_y_pred = MLP.predict(X_test)
print("\nMean squared error for Multilayer Perceptron: {:.3f}.".format(mean_squared_error(MLP_y_pred, y_test)))

#Hidden layer sizes set to 100,100
MLP = MLPR(hidden_layer_sizes = (100,100), random_state = 37).fit(X_train, y_train)
MLP_y_pred = MLP.predict(X_test)
print("\tHidden layer size set to (100,100): {:.3f}.".format(mean_squared_error(MLP_y_pred, y_test)))


#alpha set to .3
MLP = MLPR(alpha = .3, random_state = 37).fit(X_train, y_train)
MLP_y_pred = MLP.predict(X_test)
print("\tAlpha set to .3: {:.3f}.".format(mean_squared_error(MLP_y_pred, y_test)))

#beta 1 set to .89
MLP = MLPR(beta_1 = .89, random_state = 37).fit(X_train, y_train)
MLP_y_pred = MLP.predict(X_test)
print("\tBeta 1 set to .89: {:.3f}.".format(mean_squared_error(MLP_y_pred, y_test)))

#beta 2 set to .9995
MLP = MLPR(beta_2 = .9995, random_state = 37).fit(X_train, y_train)
MLP_y_pred = MLP.predict(X_test)
print("\tBeta 2 set to .9995: {:.3f}.".format(mean_squared_error(MLP_y_pred, y_test)))

#Data Standardized
MLP = MLPR(random_state = 37).fit(X_train_scaled, y_train)
MLP_y_pred = MLP.predict(X_test_scaled)
print("\tData standardized: {:.3f}.".format(mean_squared_error(MLP_y_pred, y_test)))


#All changes applied
MLP = MLPR(hidden_layer_sizes = (100,100), alpha = .3, random_state = 37, beta_1 = .89, beta_2 = .9995).fit(X_train_scaled, y_train)
MLP_y_pred = MLP.predict(X_test_scaled)
print("\tAll changes applied: {:.3f}.".format(mean_squared_error(MLP_y_pred, y_test)))
plt.scatter(MLP_y_pred, y_test, c = 'black')  
