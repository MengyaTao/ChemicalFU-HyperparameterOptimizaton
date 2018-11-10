import sys
sys.path.append("./src") # append to system path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

x_trn = pd.read_csv('./data/descs/0408_features_training.csv',header=0)
y_trn = pd.read_csv('./data/descs/0408_targets_training.csv',header=0)
x_tst = pd.read_csv('./data/descs/0408_features_test.csv',header=0)
y_tst = pd.read_csv('./data/descs/0408_targets_test.csv',header=0)

y_trn = y_trn.as_matrix()
# reshape the Y into (number,)
r,c = y_trn.shape
y_trn = np.reshape(y_trn, (r,))

r_tst,c_tst = y_tst.shape
y_tst = np.reshape(y_tst, (r_tst,))

this_scaler = StandardScaler()
x_trn = this_scaler.fit_transform(x_trn)
# x_val = this_scaler.transform(x_val)
x_tst = this_scaler.transform(x_tst)


rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True)
param_grid = {
    'n_estimators':[10,20,40,80,100,120,150,170,200,300,400],
    'max_features':['auto','sqrt','log2'],
    'max_depth':[10,20,40,80,100,120,150,170,200,300,400],
    'criterion':['gini','entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
CV_rfc.fit(x_trn,y_trn)
print CV_rfc.best_estimator_


'''
################# Training 80%, Test 20%, 5-fold CV ##########################
# load data
X = pd.read_csv('./data/descs/0407_features.csv',header=0)
X = X.fillna(X.mean()) #fill NA with col mean
Y = pd.read_csv('./data/descs/0407_targets_pesticides.csv',header=0)
# Y = pd.read_csv('./data/descs/0407_targets.csv',header=0)
# this_data = data_sampler()
# this_data.sample_data(df, num_trn_each_class=600, num_test_left=20)
# convert Y from dataframe into array
Y = Y.as_matrix()
# reshape the Y into (number,)
r,c = Y.shape
Y = np.reshape(Y, (r,))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

this_scaler = StandardScaler()
X_train = this_scaler.fit_transform(X_train)
X_test = this_scaler.transform(X_test)

# build a classification task using 3 informatiove features

rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True)
param_grid = {
    'n_estimators':[10,200],
    'max_features':['auto','sqrt','log2'],
    'max_depth':[10,200],
    'criterion':['gini','entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train,y_train)
print CV_rfc.best_estimator_

# the best estimator architecture is shown below - 9 functional uses
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=200, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=200, n_jobs=-1, oob_score=True, random_state=None,
            verbose=0, warm_start=False)

rfc.fit(X_train, y_train)
y_true, y_pred = y_test, rfc.predict(X_test)
print confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))

# the best estimator architecture is shown below - 7 functional uses
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=200, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=200, n_jobs=-1, oob_score=True, random_state=None,
            verbose=0, warm_start=False)

rfc.fit(X_train, y_train)
y_true, y_pred = y_test, rfc.predict(X_test)
print confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))

'''
