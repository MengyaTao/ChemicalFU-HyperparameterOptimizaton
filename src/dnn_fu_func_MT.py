import sys
sys.path.append("./src") # append to system path
import pandas as pd
import numpy as np
import modeling_tool as mt
from make_training_data import data_sampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing


def data_load (filename):
    # load data
    df = pd.read_csv(filename, header=0)
    # sample and split data
    this_data = data_sampler()
    this_data.sample_data(df, num_test_left=30)
    trn_X = this_data.trn_data['descs']
    trn_Y = this_data.trn_data['target']
    tst_X = this_data.tst_data['descs']
    tst_Y = this_data.tst_data['target']
    # data normaliztion
    # trn_X = preprocessing.scale(trn_X)
    # tst_X = preprocessing.scale(tst_X)
    this_scaler = StandardScaler()
    trn_X = this_scaler.fit_transform(trn_X)
    tst_X = this_scaler.transform(tst_X)

    target_names = np.unique(this_data.trn_data['class'])

    num_descs = trn_X.shape[1]
    num_target = trn_Y.shape[1]

    return trn_X, trn_Y, tst_X, tst_Y, num_descs, num_target, target_names


# topology = [input_Data, 5, 5, 5, n_classes]

def dnn_model (infile, topology, EPOCH, BATCH_SIZE, ETA, ACTIVATION_FUNCTION, OPTIMIZER):
    print len(topology)
    # load data
    trn_X, trn_Y, tst_X, tst_Y, num_descs, num_target, target_names = data_load(infile)

    X = tf.placeholder(tf.float32, shape=[None, num_descs])
    y = tf.placeholder(tf.float32, shape=[None, num_target])

    layers = {}
    lcompute = []
    # lcompute = {} # layers for computation graph

    for i in range(1, len(topology)):

        layer = {'weights': tf.Variable(tf.random_normal([topology[i-1], topology[i]])),
                 'biases': tf.Variable(tf.random_normal([topology[i]]))}
        layers[i-1] = layer

        if i == 1:
            l = tf.add(tf.matmul(X, layers[i-1]['weights']), layers[i-1]['biases'])
        else:
            l = tf.add(tf.matmul(lcompute[i-2], layers[i-1]['weights']), layers[i-1]['biases'])

        if i == len(topology) - 1:
            l
        else:
            l = ACTIVATION_FUNCTION(l)
        lcompute.append(l)

    print lcompute[len(topology) -2]


    # pred = tf.argmax(l, dimension=1)

    # Define loss and optimizer

    # regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w_out)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= lcompute[len(topology)-2], labels=y))
    optimizer = OPTIMIZER(learning_rate=ETA).minimize(cost)

    init = tf.global_variables_initializer()

    # Start Training
    training_accuracy, test_accuracy = [], []
    correct_prediction = tf.equal(tf.argmax(l, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # save the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EPOCH):
            for i in range(0, len(trn_X), BATCH_SIZE):
                _, c = sess.run([optimizer, cost], feed_dict={X: trn_X[i:i + BATCH_SIZE], y: trn_Y[i:i + BATCH_SIZE]})
            trn_acc = accuracy.eval(feed_dict={X: trn_X, y: trn_Y})
            tst_acc = accuracy.eval(feed_dict={X: tst_X, y: tst_Y})
            training_accuracy.append(trn_acc*100.)
            test_accuracy.append(tst_acc*100.)
            print("Epoch = %d, Training Accuracy = %.2f%%, Testing Accuracy = %.2f%%" % (
                epoch + 1, 100. * trn_acc, 100. * tst_acc))

    return test_accuracy

        # final_pred = sess.run(pred, feed_dict={X: tst_X, y: tst_Y})
        # plt.plot(costs)
        # plt.show()
        # print classification_report(np.argmax(tst_Y, axis=1), final_pred, target_names=target_names)
        # print confusion_matrix(np.argmax(tst_Y, axis=1), final_pred)
        # save_path = saver.save(sess, "./net/"+ outfile + ".ckpt")
        # print("Model saved in file: %s" % save_path)


