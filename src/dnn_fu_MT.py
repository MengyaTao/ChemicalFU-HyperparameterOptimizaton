import sys
sys.path.append("./src") # append to system path

import json
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

#static parameters
BATCH_SIZE = 1
BETA = 0.001

# load data
df = pd.read_csv('./data/descs/0315_10_functional_use_descs.csv',header=0)

# sample and split data
this_data = data_sampler()
this_data.sample_data(df, num_test_left=30)

trn_X = this_data.trn_data['descs']
trn_Y = this_data.trn_data['target']
N,M = trn_X.shape

tst_X = this_data.tst_data['descs']
tst_Y = this_data.tst_data['target']
N_tst, M_tst = tst_X.shape

# data normaliztion
trn_X = preprocessing.scale(trn_X)
tst_X = preprocessing.scale(tst_X)

target_names = np.unique(this_data.trn_data['class'])

from collections import Counter
print Counter(this_data.trn_data['class'])
print Counter(this_data.tst_data['class'])

this_scaler = StandardScaler()
trn_X = this_scaler.fit_transform(trn_X)
tst_X = this_scaler.transform(tst_X)

# print trn_X

def init_weights(shape):
    weights = tf.random_normal(shape,stddev = 0.1)
    return tf.Variable(weights)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

num_descs = trn_X.shape[1]
num_target = trn_Y.shape[1]

print num_descs,num_target

X = tf.placeholder(tf.float32,shape=[None,num_descs])
y = tf.placeholder(tf.float32,shape=[None,num_target])

n_nodes_hl1 = 1000
n_nodes_hl2 = 500
n_nodes_hl3 = 50

#First layer
w1 = init_weights((num_descs,n_nodes_hl1))
b1 = bias_variable([n_nodes_hl1])
l1 = tf.add(tf.matmul(X,w1),b1)
l1 = tf.nn.relu(l1)

# Second layer
w2 = init_weights((n_nodes_hl1,n_nodes_hl2))
b2 = bias_variable([n_nodes_hl2])
l2 = tf.add(tf.matmul(l1,w2),b2)
l2 = tf.nn.relu(l2)

# third layer
w3 = init_weights((n_nodes_hl2,n_nodes_hl3))
b3 = bias_variable([n_nodes_hl3])
l3 = tf.add(tf.matmul(l2,w3),b3)
l3 = tf.nn.relu(l3)

#Output layer
w_out = init_weights((n_nodes_hl3,num_target))
b_out = bias_variable([num_target])
l_out = tf.matmul(l3,w_out) + b_out

pred = tf.argmax(l_out,dimension=1)

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l_out,labels=y))

# Loss function using L2 Regularization
# regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + \
              # tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(b3)
# cost = tf.reduce_mean(cost + BETA * regularizers)

# drop out layer?
# relu_layer_dropout = tf.nn.dropout(relu_layer, keep_prob)

# try to use exponential decay of the learning rate
# GradientDescentOptimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost)
# global_step = tf.Variable(0)  # count the number of steps taken.
# start_learning_rate = 0.01
# learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()

# Start Training
costs = []
correct_prediction = tf.equal(tf.argmax(l_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# save the model
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for i in range(0, len(trn_X), BATCH_SIZE):
            _, c = sess.run([optimizer, cost], feed_dict={X: trn_X[i:i + BATCH_SIZE], y: trn_Y[i:i + BATCH_SIZE]})
        trn_acc = accuracy.eval(feed_dict={X: trn_X, y: trn_Y})
        tst_acc = accuracy.eval(feed_dict={X: tst_X, y: tst_Y})
        costs.append(tst_acc)
        print("Epoch = %d, Training Accuracy = %.2f%%, Testing Accuracy = %.2f%%" % (
        epoch + 1, 100. * trn_acc, 100. * tst_acc))

    final_pred = sess.run(pred, feed_dict={X: tst_X, y: tst_Y})
    plt.plot(costs)
    plt.show()
    print classification_report(np.argmax(tst_Y, axis=1), final_pred, target_names=target_names)
    print confusion_matrix(np.argmax(tst_Y, axis=1), final_pred)

    save_path = saver.save(sess, "./net/functional_use_classifier_nine_class_Mar15.ckpt")
    print("Model saved in file: %s" % save_path)

