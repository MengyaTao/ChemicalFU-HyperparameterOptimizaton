import sys
import os.path
# import tensorflow.contrib.skflow as learn
import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn import metrics
import pandas as pd
from make_training_data import data_sampler
from dnn_fu_func_MT import data_load
import dnn_fu_parameter_tunning as para_tunning

trn_X, trn_Y, tst_X, tst_Y, num_descs, num_target, target_names = data_load("0315_10_functional_use_descs.csv")

def run_model (classifier, logdir=None, monitor=None):
    classifier.fit(trn_X, trn_Y, logdir=logdir, monitor=monitor)
    # evaluate on the test data
    predictions = classifier.predict(tst_X)
    score = metrics.accuracy_score(tst_Y, predictions)
    return score

def instantiateModel (X, y, hyperparams):




for i in range(2):

    hyperparams = para_tunning.getHyperparameters(tune=True)
    print(hyperparams)
    classifier, monitor = instantiateModel (hyperparams)

    score = run_model(classifier, monitor=monitor)
    print("accuracy: %f" % score)

    # don't need to log this array
    del hyperparams['HIDDEN_UNITS']

    # add the test accuracy to the dict
    hyperparams['test_Accuracy'] = score

    # convert the dict to a dataframe
    log = pd.DataFrame(hyperparams, index=[0])
    print log

    # write to a csv file
    csvName = 'model_log.csv'
    if not (os.path.exists(csvName)):
        # first run, write headers
        log.to_csv('model_log.csv', mode='a')
    else:
        log.to_csv('model_log.csv', mode='a', header=False)

