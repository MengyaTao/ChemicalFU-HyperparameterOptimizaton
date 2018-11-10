import sys
sys.path.append("./src") # append to system path

import json
import pandas as pd
import numpy as np

# import modeling_tool as mt
# from make_training_data import data_sampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optunity
import optunity.metrics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# load data
data = pd.read_csv('./data/descs/0407_features.csv',header=0)
data = data.fillna(data.mean()) #fill NA with col mean
labels = pd.read_csv('./data/descs/0407_targets.csv',header=0)

# convert Y from dataframe into array
labels = labels.as_matrix()
# reshape the Y into (number,)
r,c = labels.shape
labels = np.reshape(labels, (r,))

# y = column_or_1d(y, warn=True)
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,random_state=0)

this_scaler = StandardScaler()
data = this_scaler.fit_transform(data)
# X_test = this_scaler.transform(x_test)

# for the SVM model, optunity will optimize the kernel family, choose from linear, polynomical and RBF
def train_svm(x_train, y_train, kernel, C, gamma, degree, coef0):
    if kernel == 'linear':
        model = SVC(kernel=kernel, C=C)
    elif kernel == 'poly':
        model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
    elif kernel == 'rbf':
        model = SVC(kernel=kernel, C=C, gamma=gamma)
    else:
        print("Unknown kernel function: %s" % kernel)
    model.fit(x_train, y_train)
    return model

search = {
    'algorithm':{'SVM':{'kernel':{'linear': {'C':[0,2]},
                                 'rbf': {'gamma':[0,1], 'C':[0,10]},
                                 'poly':{'degree':[2,5], 'C':[0,50], 'coef0':[0,1]}
                                 }
                       },
                 'random-forest':{'n_estimators':[10,30],
                                  'max_features':[5,20]}
                 }
}

# choose the best model based on the area under the ROC curve in 5-fold cross-validation
@optunity.cross_validated(x=data, y=labels, num_folds=5)

def performance(x_train, y_train, x_test, y_test,
                algorithm, n_estimators=None, max_features=None,
                kernel=None, C=None, gamma=None, degree=None, coef0=None):
    # fit the model
    if algorithm == 'SVM':
        model = train_svm(x_train, y_train, kernel, C, gamma, degree, coef0)
        model.fit(x_train, y_train)
    elif algorithm == 'random-forest':
        model = RandomForestClassifier(n_estimators=int(n_estimators), max_features=int(max_features))
        model.fit(x_train, y_train)
    else:
        print ("Unknown algorithm: %s" % algorithm)

    # predict the test set
    if algorithm == 'SVM':
        predictions = model.decision_function(x_test)
    else:
        predictions = model.predict_proba(x_test)[:,1]

    return optunity.metrics.roc_auc(y_test, predictions, positive=True)

# test the code with one run first
performance(algorithm='SVM',kernel='linear', C=1)

'''
# make the batch run
optimal_configuration, info, _ = optunity.maximize_structured(performance,
                                                              search_space=search,
                                                              num_evals=2)
print(optimal_configuration)
print(info.optimum)

# make the results readable, remove all Nones
solution = dict([(k,v) for k, v in optimal_configuration.items() if v is not None])
print ('Soltuion\n===========')
print ("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))

'''




