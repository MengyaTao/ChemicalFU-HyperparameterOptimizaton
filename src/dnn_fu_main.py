
# Constants
# LEARNING_RATES = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# LEARNING_RATES = [1e-9, 1e-8]

# BETA = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# BETA = [1e-6, 1e-5]


# NEURONS = [[10, 5, 2], [100, 50, 25]]
# COLORS = ['blue', 'red', 'orange', 'green','purple','black','yellow','cyan','chocolate','magenta']
# COLORS1 = ['blue', 'red', 'orange', 'green','purple', 'chocolate', 'magenta', 'cyan']
# COLORS1 = ['blue', 'red']
# NUM_EPOCHS = 400
# DESIGN = [500, 250, 125]


import dnn_fu_parameter_tunning as para_tunning
import dnn_fu_func_MT as dnn
import pandas as pd
import os
import tensorflow as tf
import csv

# learning rate
# run_dnn_eta(design, epoch, batch_size, beta, LEARNING_RATES)
# make_plot_eta(LEARNING_RATES, COLORS1)
# para_tunning.run_dnn_eta(DESIGN, NUM_EPOCHS, 10, 0.0001, LEARNING_RATES)
# para_tunning.make_plot_eta(LEARNING_RATES, COLORS1, NUM_EPOCHS)

### eta between [1e-5, 1e-4] are the best

# regularization beta
# para_tunning.run_dnn_beta(DESIGN, NUM_EPOCHS, 10, BETA, 0.001)
# para_tunning.make_plot_beta(BETA, COLORS1, NUM_EPOCHS)

### beta between [1e-6, 1e-5]?

# batch_size
# para_tunning.run_dnn_batch_size(DESIGN, NUM_EPOCHS, BATCH_SIZE, 1e-3, 1e-4)
# para_tunning.make_plot_batch_size(BATCH_SIZE, COLORS1, NUM_EPOCHS)

### batch_size between [10, 50] are good

# neural network
# para_tunning.run_dnn_architecture(NEURONS, NUM_EPOCHS, 100, 1e-5, 1e-5)
# para_tunning.make_plot_architecture(NEURONS, COLORS1, NUM_EPOCHS)

# topoloty = [input_Data, 5, 5, 5, n_classes]
# dnn_model (infile, topology, EPOCH, BATCH_SIZE, ETA, ACTIVATION_FUNCTION, OPTIMIZER)
# def dnn_model (trn_val_file, tst_file, design, EPOCH, BATCH_SIZE, BETA, ETA, ACTIVATION_FUNCTION, OPTIMIZER):
# accuracy_test = dnn.dnn_model ("0315_10_functional_use_descs.csv", [800,400,200], 350, 70, 0.000227, 0.000159, tf.train.AdamOptimizer)
'''
    hyperparams = {
        'BATCH_SIZE': batch_size,
        # 'STEPS': random.randrange(50, 500, step=50),
        'LEARNING_RATE': random.choice([1e-6, 1e-5, 1e-4, 3e-3, 6e-4, 1e-3, 1e-2]),
        # 'LEARNING_RATE': random.uniform(1e-3,5e-3),
        'OPTIMIZER': random.choice([tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer]),
        'HIDDEN_UNITS': random.choice(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]),
        'BETA': random.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        # 'BETA': random.uniform(1e-4, 1e-3),
        'ACTIVATION_FUNCTION': random.choice([tf.nn.relu, tf.nn.softsign])
    }
'''
# LEARNING_RATES = [1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3,4e-3,6e-3,8e-3,1e-2]
LEARNING_RATES = [0.0005, 0.0001, 0.0008, 0.001, 0.005, 0.01,0.05]
# LEARNING_RATES = 0.0003
# BETA = 1E-4
BETA = [1e-7,5e-07,5e-06,5e-5,5e-4]
# BATCH_SIZE = [1,10,20,50,100,200,300]
# NEURONS = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]
OPTIMIZER = tf.train.AdamOptimizer
ACTIVATION_FUNCTION = tf.nn.relu
NEURONS = [20, 30]

'''
params = {}
validation_accuracy, test_accuracy = dnn.dnn_model(
    "./data/descs/0409_all_noNA_shuffle_pesticides.csv",
    design=30, EPOCH=500, BATCH_SIZE=50,
    BETA=5e-05, ETA=0.002, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
    OPTIMIZER=OPTIMIZER)
params['Validation_ACCURACY'] = validation_accuracy
params['Test_ACCURACY'] = test_accuracy
params['NEURON'] = NEURONS
params['ETA'] = 0.005
params['BETA'] = 5e-05
params['OPTIMIZER'] = OPTIMIZER
params['ACT_FUN'] = ACTIVATION_FUNCTION
# convert the dict to a dataframe
# log = pd.DataFrame(hyperparams)
log = pd.Series(params)
print log
csvName = 'model_results_0423_tst20_pesticides.csv'
log.to_csv(csvName, mode='a')
'''

'''
three kinds of pesticides - parameter settings: (random state = 1234)
- design=40, EPOCH=130, BATCH_SIZE=50, BETA=0.0005, ETA=0.001,

one lump pesticides - parameter settings: (random state = 1234)
- design=30, EPOCH=100, BATCH_SIZE=50,
    BETA=5e-05, ETA=0.002, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
    OPTIMIZER=OPTIMIZER) - better one
- design=30, EPOCH=80, BATCH_SIZE=50,BETA=5e-05, ETA=0.002, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
'''
for neuron in NEURONS:
    for eta in LEARNING_RATES:
        for beta in BETA:
            # for optimizer in OPTIMIZER:
                # for fun in ACTIVATION_FUNCTION:
            params = {}
            validation_accuracy, test_accuracy = dnn.dnn_model(
                "./data/descs/0409_all_noNA_shuffle_pesticides.csv",
                design=neuron, EPOCH=1000, BATCH_SIZE=20,
                BETA=beta, ETA=eta, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
                OPTIMIZER=OPTIMIZER)
            params['Validation_ACCURACY'] = validation_accuracy
            params['Test_ACCURACY'] = test_accuracy
            params['NEURON'] = neuron
            params['ETA'] = eta
            params['BETA'] = beta
            params['OPTIMIZER'] = OPTIMIZER
            params['ACT_FUN'] = ACTIVATION_FUNCTION
            # convert the dict to a dataframe
            # log = pd.DataFrame(hyperparams)
            log = pd.Series(params)
            print log
            csvName = 'model_results_0423_tst20_pesticidesI.csv'
            log.to_csv(csvName, mode='a')

'''
for neuron in NEURONS:
    for eta in LEARNING_RATES:
        for beta in BETA:
            # for optimizer in OPTIMIZER:
                # for fun in ACTIVATION_FUNCTION:
            params = {}
            validation_accuracy, test_accuracy = dnn.dnn_model(
                "./data/descs/0409_all_noNA_shuffle_pesticides.csv",
                30, EPOCH=500, BATCH_SIZE=100,
                BETA=beta, ETA=eta, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
                OPTIMIZER=OPTIMIZER)
            params['Validation_ACCURACY'] = validation_accuracy
            params['Test_ACCURACY'] = test_accuracy
            params['NEURON'] = neuron
            params['ETA'] = eta
            params['BETA'] = beta
            params['OPTIMIZER'] = OPTIMIZER
            params['ACT_FUN'] = ACTIVATION_FUNCTION
            # convert the dict to a dataframe
            # log = pd.DataFrame(hyperparams)
            log = pd.Series(params)
            print log
            csvName = 'model_results_0419_tst15perc.csv'
            log.to_csv(csvName, mode='a')

for i in range(500):

    # hyperparams = para_tunning.getHyperparameters(tune=True)
    # print(hyperparams)



    validation_accuracy, test_accuracy = dnn.dnn_model("./data/descs/0409_all_noNA_shuffle_pesticides.csv",
                                  hyperparams['HIDDEN_UNITS'],EPOCH=1000, BATCH_SIZE=hyperparams['BATCH_SIZE'],BETA=hyperparams['BETA'],
                                                       ETA=hyperparams['LEARNING_RATE'], ACTIVATION_FUNCTION=hyperparams['ACTIVATION_FUNCTION'],
                                                       OPTIMIZER=hyperparams['OPTIMIZER'])

    # hyperparams['Training_ACCURACY'] = training_accuracy
    hyperparams['Validation_ACCURACY'] = validation_accuracy
    hyperparams['Test_ACCURACY'] = test_accuracy
    hyperparams['INDEX'] = i
    # convert the dict to a dataframe
    # log = pd.DataFrame(hyperparams)
    log = pd.Series(hyperparams)
    # log = pd.DataFrame(hyperparams, dtype='convert_object')
    # log = pd.DataFrame(hyperparams([(k, v) for k, v in hyperparams.iteritems()]))
    print log

    csvName = 'model_results_0415_tst15perc_pesticides.csv'
    log.to_csv(csvName, mode='a')
    print i




        with open(csvName, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in hyperparams.items():
                csv_writer.writerow([key, value])
'''
