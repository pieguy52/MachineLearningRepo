"""
Created on Wed Feb 21 16:33:52 2018
@author: victo
"""
from utils import read_data
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1 

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255.
print(X_people)
# split the data in training and test set
X_train_people, X_test_people, y_train_people, y_test_people = train_test_split(X_people, y_people, test_size = .20, stratify=y_people, random_state=37)

Features, Labels = read_data("energy.txt")
X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(Features, Labels, test_size = .20, random_state=37)

X,Y = read_data("mnist.txt")
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X, Y, test_size = .20, random_state=37)
print("read data")
"""
##############################Preprocessing##############################
"""
#normalization
from sklearn.preprocessing import Normalizer
normalizer = Normalizer().fit(X_train_people)
X_train_people_norm = normalizer.transform(X_train_people)
X_test_people_norm = normalizer.transform(X_test_people)

normalizer = Normalizer().fit(X_train_energy)
X_train_energy_norm = normalizer.transform(X_train_energy)
X_test_energy_norm = normalizer.transform(X_test_energy)

normalizer = Normalizer().fit(X_train_mnist)
X_train_mnist_norm = normalizer.transform(X_train_mnist)
X_test_mnist_norm = normalizer.transform(X_test_mnist)
print('norm')

#standardization
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler().fit(X_train_people)
X_train_people_stand = standardizer.transform(X_train_people)
X_test_people_stand = standardizer.transform(X_test_people)

standardizer = StandardScaler().fit(X_train_energy)
X_train_energy_stand = standardizer.transform(X_train_energy)
X_test_energy_stand = standardizer.transform(X_test_energy)

standardizer = StandardScaler().fit(X_train_mnist)
X_train_mnist_stand = standardizer.transform(X_train_mnist)
X_test_mnist_stand = standardizer.transform(X_test_mnist)
print('stand')

#principle component analysis
from sklearn.decomposition import PCA
pca_decomp = PCA(svd_solver = 'randomized', random_state = 37).fit(X_train_people)
X_train_people_pca = pca_decomp.transform(X_train_people)
X_test_people_pca = pca_decomp.transform(X_test_people)

pca_decomp = PCA(whiten = True, random_state = 37).fit(X_train_energy)
X_train_energy_pca = pca_decomp.transform(X_train_energy)
X_test_energy_pca = pca_decomp.transform(X_test_energy)

pca_decomp = PCA(svd_solver = 'randomized', random_state = 37).fit(X_train_mnist)
X_train_mnist_pca = pca_decomp.transform(X_train_mnist)
X_test_mnist_pca = pca_decomp.transform(X_test_mnist)
print('pca')

#non-negative matrix factorization
from sklearn.decomposition import NMF
nmf_decomp = NMF(n_components = 100, init = 'random', solver = 'mu', random_state = 37).fit(X_train_people)
X_train_people_nmf = nmf_decomp.transform(X_train_people)
X_test_people_nmf = nmf_decomp.transform(X_test_people)

nmf_decomp = NMF(n_components = 5, init = 'nndsvd', random_state = 37).fit(X_train_energy)
X_train_energy_nmf = nmf_decomp.transform(X_train_energy)
X_test_energy_nmf = nmf_decomp.transform(X_test_energy)

nmf_decomp = NMF(n_components = 100, random_state = 37).fit(X_train_mnist)
X_train_mnist_nmf = nmf_decomp.transform(X_train_mnist)
X_test_mnist_nmf = nmf_decomp.transform(X_test_mnist)
print('nmf')

#k-Means Clustering
from sklearn.cluster import KMeans as KM
km_cluster = KM(n_clusters = 10, random_state = 37).fit(X_train_people)
X_train_people_km = km_cluster.transform(X_train_people)
X_test_people_km = km_cluster.transform(X_test_people)

km_cluster = KM(n_clusters = 10, n_init = 5, random_state = 37).fit(X_train_energy)
X_train_energy_km = km_cluster.transform(X_train_energy)
X_test_energy_km = km_cluster.transform(X_test_energy)

km_cluster = KM(n_clusters = 10, random_state = 37).fit(X_train_mnist)
X_train_mnist_km = km_cluster.transform(X_train_mnist)
X_test_mnist_km = km_cluster.transform(X_test_mnist)
print('km\n')
"""
##############################Classification##############################
"""
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC
#People DONE#
###DONE###
knc_people = KNC(n_neighbors = 10, weights = 'distance', p = 1).fit(X_train_people_nmf, y_train_people)
print("Test set score of kNN people: {:.3f}".format(knc_people.score(X_test_people_nmf, y_test_people)))

###DONE###
dtc_people = DTC(random_state = 37).fit(X_train_people_stand, y_train_people)
print("Test set score of DTC people: {:.3f}".format(dtc_people.score(X_test_people_stand, y_test_people)))

###DONE###
rfc_people = RFC(n_estimators = 100, max_depth = 25, bootstrap = False, random_state = 37).fit(X_train_people_norm, y_train_people)
print("Test set score of RFC people: {:.3f}".format(rfc_people.score(X_test_people_norm, y_test_people)))

svc_people = SVC(C = 4, kernel = 'linear', random_state = 37).fit(X_train_people_stand, y_train_people)
print("Test set score of SVC people: {:.3f}".format(svc_people.score(X_test_people_stand, y_test_people)))

mlpc_people = MLPC(alpha = .1, random_state = 37).fit(X_train_people_nmf, y_train_people)
print("Test set score of MLPC people: {:.3f}".format(mlpc_people.score(X_test_people_nmf, y_test_people)))

print('people\n')
#Mnist
###DONE###
knc_mnist = KNC(weights = 'distance').fit(X_train_mnist_norm, y_train_mnist)
print("Test set score of kNN mnist: {:.3f}".format(knc_mnist.score(X_test_mnist_norm, y_test_mnist)))

###DONE###
dtc_mnist = DTC(criterion = 'entropy', max_depth = 15, min_samples_split = 3, random_state = 37).fit(X_train_mnist_nmf, y_train_mnist)
print("Test set score of DTC mnist: {:.3f}".format(dtc_mnist.score(X_test_mnist_nmf, y_test_mnist)))

###DONE###
rfc_mnist = RFC(n_estimators = 50, max_depth = 19, random_state = 37).fit(X_train_mnist_stand, y_train_mnist)
print("Test set score of RFC mnist: {:.3f}".format(rfc_mnist.score(X_test_mnist_stand, y_test_mnist)))

###DONE###
svc_mnist = SVC(C = 3, random_state = 37).fit(X_train_mnist_nmf, y_train_mnist)
print("Test set score of SVC mnist: {:.3f}".format(svc_mnist.score(X_test_mnist_nmf, y_test_mnist)))

mlpc_mnist = MLPC(hidden_layer_sizes = (200),activation = 'tanh', random_state = 37).fit(X_train_mnist_nmf, y_train_mnist)
print("Test set score of MLPC mnist: {:.3f}".format(mlpc_mnist.score(X_test_mnist_nmf, y_test_mnist)))

print('mnist\n')
"""
##############################Regression##############################
"""
from sklearn.metrics import mean_squared_error

#k-Neighbors Regressor
from sklearn.neighbors import KNeighborsRegressor as KNR
knr_energy = KNR(weights = 'distance').fit(X_train_energy_pca, y_train_energy)
y_pred_knr = knr_energy.predict(X_test_energy_pca)
print("Mean squared error for kNN: {:.3f}.".format(mean_squared_error(y_pred_knr,y_test_energy)))

#Decision Tree Regressor 
from sklearn.tree import DecisionTreeRegressor as DTR
dtr_energy = DTR(max_depth = 11, min_samples_split = 16, min_samples_leaf = 2, random_state = 37).fit(X_train_energy_stand, y_train_energy)
y_pred_dtr = dtr_energy.predict(X_test_energy_stand)
print("Mean squared error for DTR: {:.3f}.".format(mean_squared_error(y_pred_dtr,y_test_energy)))

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor as RFR
rfr_energy = RFR(n_estimators = 100, min_samples_leaf = 2, max_leaf_nodes = 1000, random_state = 37).fit(X_train_energy, y_train_energy)
y_pred_rfr = rfr_energy.predict(X_test_energy)
print("Mean squared error for RFR: {:.3f}.".format(mean_squared_error(y_pred_rfr,y_test_energy)))

#Support Vector 
from sklearn.svm import SVR
svr_energy = SVR().fit(X_train_energy_stand, y_train_energy)
y_pred_svr = svr_energy.predict(X_test_energy_stand)
print("Mean squared error for SVR: {:.3f}.".format(mean_squared_error(y_pred_svr,y_test_energy)))

from sklearn.neural_network import MLPRegressor as MLPR
mlpr_energy = MLPR(hidden_layer_sizes = (100,100), alpha = .3, random_state = 37, beta_1 = .89, beta_2 = .9995).fit(X_train_energy_stand, y_train_energy)
y_pred_mlpr = mlpr_energy.predict(X_test_energy_stand)
print("Mean squared error for MLPR: {:.3f}.".format(mean_squared_error(y_pred_mlpr,y_test_energy)))
print('energy')