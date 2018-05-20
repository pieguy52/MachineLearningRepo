import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.pipeline import make_pipeline as mp
import pandas as pd

rnafolding_X_train, rnafolding_X_test, rnafolding_y_train, rnafolding_y_test = train_test_split(np.load('rnafolding_X.npy'), np.load('rnafolding_y.npy').ravel().astype(float), test_size = .2, random_state = 37)

mol = np.load('molecules.npy')
no_mol = np.load('no_mol.npy')

x, y, z = mol.shape
indicies = np.vstack(np.unravel_index(np.arange(x*y), (y,x))).T
mol = np.hstack((mol.reshape(x*y, z), indicies))

x, y, z = no_mol.shape
indicies = np.vstack(np.unravel_index(np.arange(x*y), (y,x))).T
no_mol = np.hstack((no_mol.reshape(x*y, z), indicies))

mol_class = np.ones(mol.shape[0])
no_mol_class = np.zeros(no_mol.shape[0])

X = np.append(mol, no_mol, axis = 0)
y = np.append(mol_class, no_mol_class, axis = 0)

X = X.astype(float)

mol_X_train, mol_X_test, mol_y_train, mol_y_test = train_test_split(X, y, test_size = .2, random_state = 47)
"""
#####kNN pipes#####
"""
knn_norm_pipe = mp(MinMaxScaler(), KNC())
knn_stand_pipe = mp(StandardScaler(), KNC())
knn_pca_pipe = mp(PCA(), KNC())

"""
#####RFC pipes#####
"""
rfc_norm_pipe = mp(MinMaxScaler(), RFC(random_state = 47))
rfc_stand_pipe = mp(StandardScaler(), RFC(random_state = 47))
rfc_pca_pipe = mp(PCA(), RFC())

"""
#####SVC pipes#####
"""
svc_norm_pipe = mp(MinMaxScaler(), SVC())
svc_stand_pipe = mp(StandardScaler(), SVC())
svc_pca_pipe = mp(PCA(), SVC())

"""
#####MLPC pipes#####
"""
mlpc_norm_pipe = mp(MinMaxScaler(), MLPC(random_state = 47))
mlpc_stand_pipe = mp(StandardScaler(), MLPC(random_state = 47))
mlpc_pca_pipe = mp(PCA(), MLPC())

"""
#####kNN grid#####
"""
kNN_param_grid = {'kneighborsclassifier__n_neighbors' : [1, 2, 3, 4, 5], 'kneighborsclassifier__weights' : ['uniform', 'distance'], 'kneighborsclassifier__p' : [1, 2, 3]}

"""
Test set score: 0.12
Best parameters: {'kneighborsclassifier__n_neighbors': 1, 'kneighborsclassifier__p': 1, 'kneighborsclassifier__weights': 'uniform'}
"""
kNN_norm_grid = GSCV(knn_norm_pipe, kNN_param_grid, scoring = 'f1', cv = 5)
kNN_norm_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(kNN_norm_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(kNN_norm_grid.best_params_))
kNN_norm_results = pd.DataFrame(kNN_norm_grid.cv_results_)
display(kNN_norm_results.head)

"""
Test set score: 0.11
Best parameters: {'kneighborsclassifier__n_neighbors': 1, 'kneighborsclassifier__p': 1, 'kneighborsclassifier__weights': 'uniform'}
"""
kNN_stand_grid = GSCV(knn_stand_pipe, kNN_param_grid, scoring = 'f1', cv = 5)
kNN_stand_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(kNN_stand_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(kNN_stand_grid.best_params_))
kNN_stand_results = pd.DataFrame(kNN_stand_grid.cv_results_)
display(kNN_stand_results.head)
"""
#####rfc grid#####
"""
rf_param_grid = {'randomforestclassifier__n_estimators' : [10, 50, 100], 'randomforestclassifier__min_samples_split': range(2,5), "randomforestclassifier__min_samples_leaf" : range(1,4)}

"""
Test set score: 0.00
Best parameters: {'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__min_samples_split': 4, 'randomforestclassifier__n_estimators': 10}
"""
rf_norm_grid = GSCV(rfc_norm_pipe, rf_param_grid, scoring = 'f1', cv = 5)
rf_norm_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(rf_norm_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(rf_norm_grid.best_params_))
rf_norm_results = pd.DataFrame(rf_norm_grid.cv_results_)
display(rf_norm_results.head)
"""
Test set score: 0.00
Best parameters: {'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__min_samples_split': 4, 'randomforestclassifier__n_estimators': 10}
"""
rf_stand_grid = GSCV(rfc_stand_pipe, rf_param_grid, scoring = 'f1', cv = 5)
rf_stand_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(rf_stand_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(rf_stand_grid.best_params_))
rf_stand_results = pd.DataFrame(rf_stand_grid.cv_results_)
display(rf_stand_results.head)
"""
#####svc grid#####
"""
svc_param_grid = {"svc__C": [.001, .01, .1, 1, 10, 100], "svc__gamma" : [.001, .01, .1, 1, 10, 100]}

"""
Test set score: 0.04
Best parameters: {'svc__C': 10, 'svc__gamma': 1}
"""
svc_norm_grid = GSCV(svc_norm_pipe, svc_param_grid, scoring = "f1", cv = 5)
svc_norm_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(svc_norm_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(svc_norm_grid.best_params_))
svc_norm_results = pd.DataFrame(svc_norm_grid.cv_results_)
display(svc_norm_results.head)
"""
Test set score: 0.09
Best parameters: {'svc__C': 100, 'svc__gamma': 0.01}
"""
svc_stand_grid = GSCV(svc_stand_pipe, svc_param_grid, scoring = "f1", cv = 5)
svc_stand_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(svc_stand_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(svc_stand_grid.best_params_))
svc_stand_results = pd.DataFrame(svc_stand_grid.cv_results_)
display(svc_stand_results.head)
"""
#####mlpc grid#####
"""
mlp_param_grid = {'mlpclassifier__hidden_layer_sizes': [10, 100, (10, 10), (100,100)], 'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'], 'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam']}

"""
Test set score: 0.02
Best parameters: {'mlpclassifier__activation': 'tanh', 'mlpclassifier__hidden_layer_sizes': 10, 'mlpclassifier__solver': 'lbfgs'}
"""
mlp_norm_grid = GSCV(mlpc_norm_pipe, mlp_param_grid, scoring = 'f1', cv = 5)
mlp_norm_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(mlp_norm_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(mlp_norm_grid.best_params_))
mlp_norm_results = pd.DataFrame(mlp_norm_grid.cv_results_)
display(mlp_norm_results.head)
"""
Test set score: 0.10
Best parameters: {'mlpclassifier__activation': 'relu', 'mlpclassifier__hidden_layer_sizes': 100, 'mlpclassifier__solver': 'lbfgs'}
"""
mlp_stand_grid = GSCV(mlpc_stand_pipe, mlp_param_grid, scoring = 'f1', cv = 5)
mlp_stand_grid.fit(rnafolding_X_train, rnafolding_y_train)
print("Test set score: {:.2f}".format(mlp_stand_grid.score(rnafolding_X_test, rnafolding_y_test)))
print("Best parameters: {}".format(mlp_stand_grid.best_params_))
mlp_stand_results = pd.DataFrame(mlp_stand_grid.cv_results_)
display(mlp_stand_results.head)

"""
######################################################################################################################################################################################################################################
"""
mlp_stand_grid_mol = GSCV(mlpc_stand_pipe, mlp_param_grid, scoring = 'f1', cv = 5)
mlp_stand_grid_mol.fit(mol_X_train, mol_y_train)
print("Test set score: {:.2f}".format(mlp_stand_grid_mol.score(mol_X_test, mol_y_test)))
print("Best parameters: {}".format(mlp_stand_grid_mol.best_params_))
mlp_stand_mol_results = pd.DataFrame(mlp_stand_grid_mol.cv_results_)
display(mlp_stand_mol_results.head)

svc_stand_grid_mol = GSCV(svc_stand_pipe, svc_param_grid, scoring = "f1", cv = 5)
svc_stand_grid_mol.fit(mol_X_train, mol_y_train)
print("Test set score: {:.2f}".format(svc_stand_grid_mol.score(mol_X_test, mol_y_test)))
print("Best parameters: {}".format(svc_stand_grid_mol.best_params_))

rf_stand_grid_mol = GSCV(rfc_stand_pipe, rf_param_grid, scoring = 'f1', cv = 5)
rf_stand_grid_mol.fit(mol_X_train, mol_y_train)
print("Test set score: {:.2f}".format(rf_stand_grid_mol.score(mol_X_test, mol_y_test)))
print("Best parameters: {}".format(rf_stand_grid_mol.best_params_))

kNN_stand_grid_mol = GSCV(knn_stand_pipe, kNN_param_grid, scoring = 'f1', cv = 5)
kNN_stand_grid_mol.fit(mol_X_train, mol_y_train)
print("Test set score: {:.2f}".format(kNN_stand_grid_mol.score(mol_X_test, mol_y_test)))
print("Best parameters: {}".format(kNN_stand_grid_mol.best_params_))
kNN_stand_results = pd.DataFrame(kNN_stand_grid_mol.cv_results_)
display(kNN_stand_results.head)