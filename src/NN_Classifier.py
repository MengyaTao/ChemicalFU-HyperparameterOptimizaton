import sys
sys.path.append("./src") # append to system path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# load data
x_trn = pd.read_csv('./data/descs/0409_features_training.csv',header=0)
x_tst = pd.read_csv('./data/descs/0409_features_test.csv',header=0)
# X = x_trn.fillna(x_trn.mean()) #fill NA with col mean
y_trn = pd.read_csv('./data/descs/0409_targets_training.csv',header=0)
y_tst = pd.read_csv('./data/descs/0409_targets_test.csv',header=0)

#Y = pd.read_csv('./data/descs/0407_targets.csv',header=0)
# this_data = data_sampler()
# this_data.sample_data(df, num_trn_each_class=600, num_test_left=20)
# convert Y from dataframe into array
y_trn = y_trn.as_matrix()
# reshape the Y into (number,)
r_trn,c_trn = y_trn.shape
y_trn = np.reshape(y_trn, (r_trn,))

y_tst = y_tst.as_matrix()
r_tst,c_tst = y_tst.shape
y_tst = np.reshape(y_tst, (r_tst,))
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

this_scaler = StandardScaler()
x_trn = this_scaler.fit_transform(x_trn)
x_tst = this_scaler.transform(x_tst)

'''
param_grid = {
    'activation':['relu'],
    'hidden_layer_sizes':[(300,150,75),(100,50,25),(100,50)],
    'solver':['adam'],
    'alpha':[0.00001, 0.0001, 0.001, 0.1,1,10],
    'tol':[0.001,0.01],
    'learning_rate':['adaptive']
}

CV_nn = GridSearchCV(estimator=MLPClassifier(verbose=True), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
CV_nn.fit(x_trn,y_trn)
print CV_nn.best_estimator_

'''


nn = MLPClassifier(activation='relu', alpha=10, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 50, 25), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.001, validation_fraction=0.1,
       verbose=True, warm_start=False)

nn.fit(x_trn, y_trn)
y_true, y_pred = y_tst, nn.predict(x_tst)
print confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
nn.fit.n_inter_





