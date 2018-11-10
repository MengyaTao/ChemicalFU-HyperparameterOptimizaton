
import json
import matplotlib.pyplot as plt
import numpy as np
import dnn_fu_func_MT as dnn
import random
import tensorflow as tf
'''
# Constants
LEARNING_RATES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
BETA = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
BATCH_SIZE = [1,10,20,50,100,200,300]
NEURONS = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]
# COLORS = ['blue', 'red', 'orange', 'green','purple','black','yellow','cyan','chocolate','magenta']
COLORS1 = ['blue', 'red', 'orange', 'green','purple', 'chocolate', 'magenta']
NUM_EPOCHS = 100
'''
LEARNING_RATES = [1e-6, 1e-5, 1e-4,1e-3]
NEURONS = [10,20,30,40,50,60]
BETAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
COLORS = ['blue1', 'blue2''red', 'orange', 'green','purple','black','yellow','cyan','chocolate','magenta']
# def dnn_model (infile, design, EPOCH, BATCH_SIZE, BETA, ETA)
# dnn.dnn_model("./data/descs/0409_all_noNA_shuffle_pesticides.csv",
# hyperparams['HIDDEN_UNITS'],EPOCH=2000, BATCH_SIZE=hyperparams['BATCH_SIZE'],BETA=hyperparams['BETA'],
# ETA=hyperparams['LEARNING_RATE'], ACTIVATION_FUNCTION=hyperparams['ACTIVATION_FUNCTION'],
# OPTIMIZER=hyperparams['OPTIMIZER'])
# learning rate
def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def num_rand_col(num):
    col = []
    for i in range(num):
        color = get_random_color(pastel_factor=0.5)
        # col.append(generate_new_color(col), pastel_factor=0.9)
        col.append(color)
    # print "Your colors:", col
    return col

num_rand_col(10)

def run_dnn_all(NEURONS, epoch, batch_size, BETAS, LEARNING_RATES):
    results_all = []
    for neuron in NEURONS:
        for eta in LEARNING_RATES:
            for beta in BETAS:
                print "\nTrain a network using neuron = "+str(neuron)+",eta = "+str(eta)+",beta = "+str(beta)
                results_all.append(
                    dnn.dnn_model ("./data/descs/0409_all_noNA_shuffle_pesticides.csv", neuron,EPOCH=epoch,
                                BATCH_SIZE=batch_size,BETA=beta,ETA=eta, ACTIVATION_FUNCTION=tf.nn.relu,OPTIMIZER=tf.train.AdamOptimizer))

    print results_all
    f = open("multiple_all_0415_2.json", "w")
    json.dump(results_all, f)
    f.close()

def make_plot_eta(NEURONS, LEARNING_RATES, BETAS, NUM_EPOCHS, num_col):
    f = open("multiple_all_0415_2.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    COLORS = num_rand_col(num_col)
    for neuron in NEURONS:
        for eta in LEARNING_RATES:
            for beta in BETAS:
                for result, color in zip(results, COLORS):
                    test_accuracy = result
                    ax.plot(np.arange(NUM_EPOCHS),
                            test_accuracy, "o-",label="$\ eta$ = " + str(eta) + "neuron = " + str(neuron) + "beta = " + str(beta),
                            color=color)
                    print "plot finished for this one"

    '''
        for neuron, eta, beta, result, color in zip(NEURONS,LEARNING_RATES,BETAS,results, COLORS):
        test_accuracy = result
        ax.plot(np.arange(NUM_EPOCHS), test_accuracy, "o-",
                label="$\ eta$ = "+str(eta)+ "neuron = "+str(neuron)+"beta = "+str(beta),
                color=color)
    '''

    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig('all_plot_0415_3.png')

# run_dnn_all(NEURONS, 1000, 10, BETAS, LEARNING_RATES)
# print "model run is finished makeing plots"
# make_plot_eta(NEURONS, LEARNING_RATES, BETAS, 1000, 120)

################################################################
# regularization beta
def run_dnn_beta(design, epoch, batch_size, BETA, eta):
    results_beta = []
    for beta in BETA:
        print "\nTrain a network using beta = "+str(beta)
        results_beta.append(
            dnn.dnn_model ("0315_10_functional_use_descs.csv", design,
           EPOCH=epoch, BATCH_SIZE=batch_size,BETA=beta,ETA=eta))

    f = open("multiple_beta.json", "w")
    json.dump(results_beta, f)
    f.close()

def make_plot_beta(BETA, COLORS1, NUM_EPOCHS):
    f = open("multiple_beta.json", "r")
    results_beta = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for beta, result, color in zip(BETA, results_beta, COLORS1):
        test_accuracy = result
        ax.plot(np.arange(NUM_EPOCHS), test_accuracy, "o-",
                label="$\ beta$ = "+str(beta),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Testing Accuracy')
    plt.legend(loc='lower right')
    # plt.show()
    fig.savefig('beta_plot.png')


################################################################
# batch size
def run_dnn_batch_size(design, epoch, BATCH_SIZE, beta, eta):
    results_size = []
    for size in BATCH_SIZE:
        print "\nTrain a network using batch_size = "+str(size)
        results_size.append(
            dnn.dnn_model ("0315_10_functional_use_descs.csv", design,
           EPOCH=epoch, BATCH_SIZE=size,BETA=beta,ETA=eta))

    f = open("multiple_batch_size.json", "w")
    json.dump(results_size, f)
    f.close()

def make_plot_batch_size(BATCH_SIZE, COLORS1, NUM_EPOCHS):
    f = open("multiple_batch_size.json", "r")
    results_size = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for size, result, color in zip(BATCH_SIZE, results_size, COLORS1):
        test_accuracy = result
        ax.plot(np.arange(NUM_EPOCHS), test_accuracy, "o-",
                label="$\ batch_size$ = "+str(size),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Testing Accuracy')
    plt.legend(loc='lower right')
    # plt.show()
    fig.savefig('batch_size_plot.png')

###############################################################
# nn architecture
def run_dnn_architecture(NEURONS, epoch, batch_size, beta, eta):
    results_neurons = []
    for neuron in NEURONS:
        print "\nTrain a network using neurons = "+str(neuron)
        results_neurons.append(
            dnn.dnn_model ("0315_10_functional_use_descs.csv", neuron,
           EPOCH=epoch, BATCH_SIZE=batch_size,BETA=beta,ETA=eta))

    f = open("multiple_neurons.json", "w")
    json.dump(results_neurons, f)
    f.close()

def make_plot_architecture(NEURONS, COLORS1, NUM_EPOCHS):
    f = open("multiple_neurons.json", "r")
    results_neurons = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for neuron, result, color in zip(NEURONS, results_neurons, COLORS1):
        test_accuracy = result
        ax.plot(np.arange(NUM_EPOCHS), test_accuracy, "o-",
                label="$\ neurons$ = "+str(neuron),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Testing Accuracy')
    plt.legend(loc='lower right')
    # plt.show()
    fig.savefig('neuron_plot.png')

###############################################################
# randome hyperparameter search (random sampling)
# dnn_model (infile, design, EPOCH, BATCH_SIZE, BETA, ETA):
def getHyperparameters (tune=False):
    if tune:
        # one hidden layer is sufficient for majority of problems, unless there is a reason to expand the # of hidden layers
        # we don't want to have the overfitting problem: E_trn < E_val or E_tst
        # randomize dnn layers and hidden size
        # rule of thumbs try
        # 1) between (5,200) - between input neurons number and output neurons number
        # 2) genetic-algorithm-based search uppder bound (Nh=Ns(alpha*(Ni+No)), alpha ranges [2,10], so between (1,7)
        # 3) 2/3 size of input layer + output layer, so around 130 neurons (120,140)
        # 4) less than twice of input neurons, so < 364
        # 5) geometric pyramid rules proposed in 1983, sqrt(n*m) = sqrt(9*182), so round 40, [30,50]
        # let's try out one by one then
        # method 2
        # l1_nodes = random.randrange(1, 10, step=1)
        # method 5
        # l1_nodes = random.randrange(30, 50, step=1)
        # method 3
        # l1_nodes = random.randrange(120, 140, step=1)
        # method 4 & 1
        # l1_nodes = random.randrange(5, 360, step=2)

        # after trying all of the methods; we would conclude that single layer performs as good as multiple layers for our data
        # also, fewer neurons can have high accuracy too;
        # so we would narrow down the number of neurons from [20,30], and optimize other parameters for one pesticides
        # l1_nodes = random.randrange(30, 40, step=1)

        # l2_nodes = max(int(l1_nodes*random.uniform(0.5,1)),1)
        # l3_nodes = max(int(l2_nodes*random.uniform(0.5,1)),1)
        # l4_nodes = int(l3_nodes * 0.5)
        # l5_nodes = int(l4_nodes * 0.5)


        # make dictionary of randomized hyperparameters
        # after grid search method - identify the parameter rough range
        # after random search method - thousands of trys, we have figured out that a good combination of hyperparameters
        # specifically good for our problem: AdamOptimizaer, relu activation function, range of Beta size (10,530)
        # range of learning rate: [1E-04, 9E-03], range of beta (regularization term): [1E-05, 9E-04]
        # however, those ranges can definitely be narrowed down once more with more trials and errors;

        # after trying the # of units between 20-30, I found out that 20 can give out very good comparable performances
        # so I would like to stick with 20 neurons and optimize the other parameters;
        # l1_nodes = random.randrange(70, 160, step=2)
        # batch_size = random.randrange(100, 160, step=20)
        batch_size = 10

        hyperparams = {
            'BATCH_SIZE': batch_size,
            #'STEPS': random.randrange(50, 500, step=50),
            'LEARNING_RATE': random.choice([1e-6, 1e-5, 1e-4,3e-3, 6e-4, 1e-3,1e-2]),
            #'LEARNING_RATE': random.uniform(1e-3,5e-3),
            'OPTIMIZER': random.choice([tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer]),
            'HIDDEN_UNITS': random.choice([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]),
            'BETA': random.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            #'BETA': random.uniform(1e-4, 1e-3),
            'ACTIVATION_FUNCTION': random.choice([tf.nn.relu,tf.nn.softsign])
        }


    else:
        hidden_units=[100,50,25]
        hyperparams = {
            'BATCH_SIZE': 10,
            'NUM_EPOCHS': 100,
            'LEARNING_RATE': 0.001,
            'OPTIMIZER': 'Adam',
            'HIDDEN_UNITS': hidden_units,
            'NUM_LAYERS': len(hidden_units),
            'NUM_UNITS': hidden_units[0],
            'ACTIVATION_FUNCTION': tf.nn.relu,
            'KEEP_PROB': 0.5,
            'MAX_BAD_COUNT': random.randrange(10, 1000, 10)
        }
    return hyperparams
'''
def getHyperparameters (tune=False):
    if tune:
        # randomize dnn layers and hidden size
        hidden_units=[]
        Nlayers = random.randint(1,6)
        Nunits = random.randrange(100, 1000, step=50)
        hidden_units.append(Nunits)

        for i in range(2, Nlayers+1):
            Nunits = int(hidden_units[i-2]*0.5)
            hidden_units.append(Nunits)
            i = i+1

        # make dictionary of randomized hyperparameters
        hyperparams = {
            'BATCH_SIZE': random.randrange(10, 50, step=5),
            'STEPS': random.randrange(100, 400, step=50),
            'LEARNING_RATE': random.uniform(1e-6, 1e-3),
            'OPTIMIZER': random.choice([tf.train.GradientDescentOptimizer, tf.train.AdamOptimizer, tf.train.AdadeltaOptimizer,
                                        tf.train.AdagradOptimizer,
                                        tf.train.FtrlOptimizer, tf.train.RMSPropOptimizer]),
            #'OPTIMIZER': random.choice(['GradientDescentOptimizer', 'AdamOptimizer', 'AdadeltaOptimizer',
                                        #'AdagradDAOptimizer', 'AdagradOptimizer', 'MomentumOptimizer',
                                        #'FtrlOptimizer', 'RMSPropOptimizer']),
            'HIDDEN_UNITS': hidden_units,
            'NUM_LAYERS':Nlayers,
            #'NUM_UNITS': Nunits,
            #'BETA': random.uniform(1e-6, 1e-3),
            'ACTIVATION_FUNCTION': random.choice([tf.nn.relu, tf.nn.tanh, tf.nn.relu6, tf.nn.softsign,
                                                  tf.nn.elu, tf.nn.sigmoid, tf.nn.softplus]),
            # 'DROPOUT': random.uniform(0.1, 1.0),
            # 'MAX_BAD_COUNT': random.randrange(10, 1000, 10)
        }

    else:
        hidden_units=[100,50,25]
        hyperparams = {
            'BATCH_SIZE': 10,
            'NUM_EPOCHS': 100,
            'LEARNING_RATE': 0.001,
            'OPTIMIZER': 'Adam',
            'HIDDEN_UNITS': hidden_units,
            'NUM_LAYERS': len(hidden_units),
            'NUM_UNITS': hidden_units[0],
            'ACTIVATION_FUNCTION': tf.nn.relu,
            'KEEP_PROB': 0.5,
            'MAX_BAD_COUNT': random.randrange(10, 1000, 10)
        }
    return hyperparams
'''
