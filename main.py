## Yunchuan Kong
## 2018 Copyright Reserved

from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics
from random import seed
import time
import sys

tf.reset_default_graph()
file1 = sys.argv[1]
file2 = sys.argv[2]
out_file = sys.argv[3]

## load in data
partition = np.loadtxt(file2, dtype=int, delimiter=None)
expression = np.loadtxt(file1, dtype=float, delimiter=",")
label_vec = np.array(expression[:,-1], dtype=int)
expression = np.array(expression[:,:-1])
labels = []
for l in label_vec:
    if l == 1:
        labels.append([0,1])
    else:
        labels.append([1,0])
labels = np.array(labels,dtype=int)

## train/test data split
cut = int(0.8*np.shape(expression)[0])
expression, labels = shuffle(expression, labels)
x_train = expression[:cut, :]
x_test = expression[cut:, :]
y_train = labels[:cut, :]
y_test = labels[cut:, :]

## hyper-parameters and settings
L2 = False
max_pooling = False
droph1 = False
learning_rate = 0.0001
training_epochs = 100
batch_size = 8
display_step = 1

## the constant limit for feature selection
gamma_c = 50
gamma_numerator = np.sum(partition, axis=0)
gamma_denominator = np.sum(partition, axis=0)
gamma_numerator[np.where(gamma_numerator>gamma_c)] = gamma_c


n_hidden_1 = np.shape(partition)[0]
n_hidden_2 = 64
n_hidden_3 = 16
n_classes = 2
n_features = np.shape(expression)[1]

## initiate training logs
loss_rec = np.zeros([training_epochs, 1])
training_eval = np.zeros([training_epochs, 2])

def max_pool(mat): ## input {mat}rix

    def max_pool_one(instance):
        return tf.reduce_max(tf.multiply(tf.matmul(tf.reshape(instance, [n_features, 1]), tf.ones([1, n_features]))
                                         , partition)
                             , axis=0)

    out = tf.map_fn(max_pool_one, mat, parallel_iterations=1000, swap_memory=True)
    return out

def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, tf.multiply(weights['h1'], partition)), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    if max_pooling:
        layer_1 = max_pool(layer_1)
    if droph1:
        layer_1 = tf.nn.dropout(layer_1, keep_prob=keep_prob)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    ## Do not use batch-norm
    # layer_3 = tf.contrib.layers.batch_norm(layer_3, center=True, scale=True,
    #                                   is_training=is_training)
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob=keep_prob)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

weights = {
    'h1': tf.Variable(tf.truncated_normal(shape=[n_features, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_hidden_3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_3, n_classes], stddev=0.1))

}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
if L2:
    reg = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
          tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out'])
    cost = tf.reduce_mean(cost + 0.01 * reg)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

## Evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
y_score = tf.nn.softmax(logits=pred)

var_left = tf.reduce_sum(tf.abs(tf.multiply(weights['h1'], partition)), 0)
var_right = tf.reduce_sum(tf.abs(weights['h2']), 1)
var_importance = tf.add(tf.multiply(tf.multiply(var_left, gamma_numerator), 1./gamma_denominator), var_right)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_batch = int(np.shape(x_train)[0] / batch_size)

    ## Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        x_tmp, y_tmp = shuffle(x_train, y_train)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = x_tmp[i*batch_size:i*batch_size+batch_size], \
                                y_tmp[i*batch_size:i*batch_size+batch_size]

            _, c= sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
                                                        keep_prob: 0.9,
                                                        lr: learning_rate
                                                        })
            # Compute average loss
            avg_cost += c / total_batch

        del x_tmp
        del y_tmp

        ## Display logs per epoch step
        if epoch % display_step == 0:
            loss_rec[epoch] = avg_cost
            acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_train, y: y_train, keep_prob: 1})
            auc = metrics.roc_auc_score(y_train, y_s)
            training_eval[epoch] = [acc, auc]
            print ("Epoch:", '%d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost),
                    "Training accuracy:", round(acc,3), " Training auc:", round(auc,3))

        if avg_cost <= 0.1:
            print("Early stopping.")
            break

    ## Testing cycle
    acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_test, y: y_test, keep_prob: 1})
    auc = metrics.roc_auc_score(y_test, y_s)
    var_imp = sess.run([var_importance])
    var_imp = np.reshape(var_imp, [n_features])
    print("*****=====", "Testing accuracy: ", acc, " Testing auc: ", auc, "=====*****")

np.savetxt(out_file, var_imp, delimiter=",")
