from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.cross_validation as skcv
import csv

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

training_iters = 1000000
batch_size = 500
display_step = 5
keep_prob=0.75

def write_output_file(y_test,name):
    #Write to an output file
    with open(name, 'wt') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Id','y'))
        line=[0,0.0]
        for i in range(len(y_test)):
            line[0]=45324+i
            line[1]=y_test[i]
            writer.writerow(line)
        print("File ",name," written")

#mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize."""
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, 'activation')
        return activations

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 100])
hidden1 = nn_layer(x, 100, 500, 'layer1')
hidden1_dropped = tf.nn.dropout(hidden1, keep_prob)
hidden2 = nn_layer(hidden1_dropped,500,20,'layer2', act=tf.nn.sigmoid)
hidden2_dropped = tf.nn.dropout(hidden2, keep_prob)
y = nn_layer(hidden2_dropped, 20, 5, 'layer3', act=tf.nn.softmax)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 5])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Train
init = tf.initialize_all_variables()

train = pd.read_hdf("train.h5", "train")
#print(train)
index=[]
for key in train.keys():	
    if(key!="Id" and key!="y"):
        index.append(key)

#Create X array        
X=np.array(train[index])
#print(len(X[0]))
#Create Y array
Y=np.array(train['y'])
#print(Y)
#fix array Y
y_edit=np.zeros((len(Y),5),dtype=np.int)
for i in range(len(Y)):
	y_edit[i][Y[i]]=1

print(y_edit)
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, y_edit, train_size=0.9)

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        index2 = np.random.choice(len(Xtrain), batch_size)  # using stochastic batches of the training set to train the neural network
        batch_xs = np.array(Xtrain[index2])
        batch_ys = np.array(Ytrain[index2])
        # Fit training using batch data
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if step % display_step == 0:
            # Calculate batch accuracy
            #acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            # Calculate batch loss
            #loss = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys})
            print ("Iter " + str(step*batch_size),"Testing Accuracy:", sess.run(accuracy, feed_dict={x: Xtest, y_: Ytest}))
        step += 1
    print ("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: Xtest, y_: Ytest}))

    #For test data set
    test = pd.read_hdf("test.h5", "test")
    index_test = []
    for key in test.keys(): 
        if(key!="Id" and key!="y"):
            index_test.append(key)        
    x_test=np.array(test[index_test])

    y_predict = sess.run(tf.argmax(y, 1), feed_dict={x: x_test})
    write_output_file(y_predict,"project3_relu.csv")


