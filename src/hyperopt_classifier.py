import sys
sys.path.append("./src") # append to system path

import pandas as pd
import numpy as np
import hpsklearn
from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.preprocessing import StandardScaler
import hyperopt.tpe

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


estimator = HyperoptEstimator( classifier=any_classifier('clf'), algo=hyperopt.tpe.suggest, max_evals=100)

estimator.fit( x_trn, y_trn )

print( estimator.score( x_tst, y_tst ) )
# <<show score here>>
print(estimator.best_model())
