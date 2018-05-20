import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.pipeline import make_pipeline as mp
import sklearn.metrics as metrics

def remove_char(essay):
    remove_char = ['\\r', '\\n', '!', '?',';', ':', '-', '+', '~', ',', '.', '...', '%', '$', '#', '\"', '"', '\\', '/', '(', ')']
    for char in remove_char:
        if char in essay:
            essay = essay.replace(char, ' ')
    remove_char = ["'", '\'']
    for char in remove_char:
        if char in essay:
            essay = essay.replace(char, '')
    essay = essay.lower().split()
    return essay

train_df = pd.read_csv('train.csv', sep=',')

word_vectors = {}
with open("glove.6B.50d.txt", encoding="utf8") as f:
    for line in f:
       vals = line.split()
       key = str(vals[0])
       word_vectors[key] = [float(s) for s in vals[1:]]
       
essay_1 = train_df["project_essay_1"].as_matrix()
for i in range(essay_1.shape[0]):
    essay_1[i] = np.asarray(remove_char(essay_1[i]))
    
essay_2 = train_df["project_essay_2"].as_matrix()
for i in range(essay_2.shape[0]):
    essay_2[i] = np.asarray(remove_char(essay_2[i]))

count = 0

X1 = np.array([])
for i in range(essay_1.shape[0]):
    temp = np.array([])
    for word in essay_1[i]:
        try:
            vector = word_vectors[word]
            temp = np.concatenate((temp, vector), axis = 0)
        except:
            count += 1
            pass
    temp = temp.reshape(int(temp.size/50), 50)
    temp = np.mean(temp, axis = 0)
    X1 = np.concatenate((X1, temp), axis = 0)

X2 = np.array([])
for i in range(essay_2.shape[0]):
    temp = np.array([])
    for word in essay_2[i]:
        try:
            vector = word_vectors[word]
            temp = np.concatenate((temp, vector), axis = 0)
        except:
            count += 1
            pass
    temp = temp.reshape(int(temp.size/50), 50)
    temp = np.mean(temp, axis = 0)
    X2 = np.concatenate((X2, temp), axis = 0)

print(count)

X1 = X1.reshape(int(X1.shape[0]/50), 50)
X2 = X2.reshape(int(X2.shape[0]/50), 50)
X = np.concatenate((X1, X2), axis = 1)
X = np.nan_to_num(X)

y = train_df['project_is_approved'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 47)

mlpc_norm_pipe = mp(MinMaxScaler(), MLPC(random_state = 47))

mlp_param_grid1 = {'mlpclassifier__hidden_layer_sizes': [10, 100, (10, 10), (100,100)], 'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'], 'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam']}
mlp_param_grid2 = {'hidden_layer_sizes': [10, 100, (10, 10), (100,100)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam']}

mlp_norm_grid = GSCV(mlpc_norm_pipe, mlp_param_grid1, scoring = 'f1', cv = 5)
mlp_norm_grid.fit(X_train, y_train)
print("Test set score: {:.2f}".format(mlp_norm_grid.score(X_test, y_test)))
print("Best parameters: {}".format(mlp_norm_grid.best_params_))

mlp_norm_grid = GSCV(MLPC(), mlp_param_grid2, scoring = 'f1', cv = 5)
mlp_norm_grid.fit(X_train, y_train)
print("Test set score: {:.2f}".format(mlp_norm_grid.score(X_test, y_test)))
print("Best parameters: {}".format(mlp_norm_grid.best_params_))

mlpc_results = pd.DataFrame(mlp_norm_grid.cv_results_)
display(mlpc_results.head)

y_pred = mlp_norm_grid.predict(X_test)
metrics.roc_auc_score(y_test, y_pred)
metrics.accuracy_score(y_test, y_pred)