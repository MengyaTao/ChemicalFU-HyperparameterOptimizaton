import sys
sys.path.append("./src") # append to system path

import pandas as pd
import numpy as np
from make_training_data import data_sampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pylab as pl


################# Training + Validation < 400, Test 30, 5-fold CV ##################
### the dataset was manually separately into two datasets; 1) training+validation 2) test

# load data
# X_all = pd.read_csv('./data/descs/0408_features_training_test.csv',header=0)
# X_all = X_all.fillna(X_all.mean()) #fill NA with col mean
# X_all_rmNA = X_all.to_csv("0408_features_training_test_noNA.csv", sep='\t')
x_trn = pd.read_csv('./data/descs/0408_features_training.csv',header=0)
y_trn = pd.read_csv('./data/descs/0408_targets_training.csv',header=0)
x_tst = pd.read_csv('./data/descs/0408_features_test.csv',header=0)
y_tst = pd.read_csv('./data/descs/0408_targets_test.csv',header=0)
# Y = pd.read_csv('./data/descs/0408_targets_training_pesticides.csv',header=0)
# this_data = data_sampler()
# this_data.sample_data(df, num_trn_each_class=600, num_test_left=20)
# convert Y from dataframe into array
y_trn = y_trn.as_matrix()
# reshape the Y into (number,)
r,c = y_trn.shape
y_trn = np.reshape(y_trn, (r,))

r_tst,c_tst = y_tst.shape
y_tst = np.reshape(y_tst, (r_tst,))

# x_trn, x_val, y_trn, y_val = train_test_split(x, y, test_size=0.3,random_state=0)

this_scaler = StandardScaler()
x_trn = this_scaler.fit_transform(x_trn)
# x_val = this_scaler.transform(x_val)
x_tst = this_scaler.transform(x_tst)

'''
C_range = 10. ** np.arange(-3,4)
gamma_range = 10. ** np.arange(-3, 4)

# kernel = ['linear','poly','rbf']
# use the default kernel 'rbf' first, with C_range and gamma

param_grid = dict(gamma = gamma_range, C = C_range)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(x_trn, y_trn)

print ("the best classifier is: ", grid.best_estimator_)


the best classifier is: ', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)


clf = SVC(C=10.0, cache_size=500, class_weight="balanced", coef0=0.0,
          decision_function_shape=None, degree=3, gamma=0.01, kernel='linear',
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False)
clf.fit(x_trn, y_trn)
y_true, y_pred = y_tst, clf.predict(x_tst)
print confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))

'''


param_grid = {
    'C':[0.0001,0.001,0.01,0.1,1,10,100,1000],
    'kernel':('linear','poly','rbf'),
    'gamma':[0.0001,0.001,0.01,0.1,1,10,100,1000],
    'cache_size':[500,1000],
    'coef0': [0.01,0.1,0,1,10],
    'degree':[1,2,3,4,5],
    'tol':[0.00001,0.0001,0.001,0.01,0.1],
    'class_weight':('balanced','None')
}


grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

grid.fit(x_trn, y_trn)

print ("the best classifier is: ", grid.best_estimator_)


# Grid Search for the best estimator hyperparameters
'''

X_train, X_test, y_train, y_test = train_test_split(trn_X, trn_Y, test_size=0.2,random_state=0)
this_scaler = StandardScaler()
X_train = this_scaler.fit_transform(X_train)
X_test = this_scaler.transform(X_test)


C_range = 10. ** np.arange(-3,8)
gamma_range = 10. ** np.arange(-5, 4)

param_grid = dict(gamma = gamma_range, C = C_range)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

grid.fit(X_train, y_train)

print ("the best classifier is: ", grid.best_estimator_)
'''












################# Training 80%, Test 20%, 5-fold CV ##########################


'''

# load data
X = pd.read_csv('./data/descs/0407_features.csv',header=0)
X = X.fillna(X.mean()) #fill NA with col mean
# Y = pd.read_csv('./data/descs/0407_targets_pesticides.csv',header=0)
Y = pd.read_csv('./data/descs/0407_targets.csv',header=0)
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

# for training set, get 5-fold cross validation

# clf = svm.SVC(kernel='linear', C=1, gamma=0.001)
# the "scoring" parameter, classification: 'accuracy', 'average_precision','f1','f1_micro','f1_macro',
# 'f1_weighted',
# scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
# print scores
# the best classifier is: ', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
# max_iter=-1, probability=False, random_state=None, shrinking=True,
# tol=0.001, verbose=False)


clf = SVC(C=10, kernel='rbf', gamma=0.01, tol=0.001, cache_size=200)
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)
print confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))


param_grid = [
    {'C':[1,10,100,1000], 'kernel':['linear']},
    {'C':[1,10,100,1000], 'gamma':[0.01,0.001,0.0001,0.00001], 'kernel':['rbf']}
]

scores = ['precision','recall']

for score in scores:
    print ('# Tuning hyper-parameters for %s' % score)
    print ()

    clf = GridSearchCV(SVC(C=1), param_grid, cv=5,
                       scoring='%s_macro' % score)

    clf.fit(X_train,y_train)
    print("best parameters set found on validation set:")
    print()
    print(clf.best_params_)
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print ("%0.3f (+/-%0.03f) for %r"
               % (mean, std*2, params))

        print("detailed classification report:")
        print()
        print("the model is trained on the 80% of the dataset.")
        print("the scores are computed on the 5-fold cross validation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print('the best fitted model for test set (20%) of the dataset')
        print(classification_report(y_true, y_pred))
        print('confusion matrix')
        print confusion_matrix(y_true, y_pred)
        print()


C_range = 10. ** np.arange(-3,8)
gamma_range = 10. ** np.arange(-5, 4)

param_grid = dict(gamma = gamma_range, C = C_range)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

grid.fit(X_train, y_train)

print ("the best classifier is: ", grid.best_estimator_)
'''

# exhaustive grid search


# save the model
# import pickle
# s = pickle.dump(clf)
# clf2 = pickle.loads(s)
# clf2.predict(X_test)














