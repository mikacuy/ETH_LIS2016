# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:27:54 2016

@author: mikacuy
@reader: wenr:)
"""

import csv
import tensorflow as tf
import numpy as np
import pandas

import sklearn.cross_validation as skcv
from sklearn.cross_validation import KFold
import sklearn.preprocessing as skpr
#from sklearn.neural_network import MLPClassifier

def rms(gtruth, pred):
    diff = gtruth - pred
    return np.sqrt(np.mean(np.square(diff)))
    
def get_accuracy(test,pred):
    correct=0
    for i in range(len(test)):
        if test[i]==pred[i]:
            correct+=1
    return float(correct)/float(len(test))

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


# Do k-fold cross validation on the entire training set
# and generate a score which is the average of the k-fold
# socres
def crossValidation(estimator,xtrain, ytrain, k_fold):
	kf_total = skcv.KFold(len(xtrain), n_folds=k_fold)
	score=0	
	i=0
	for train_index, test_index in kf_total:		
		i+=1	
		estimator.fit(xtrain[train_index],ytrain[train_index])
		ypredict=estimator.predict(xtrain[test_index])
		#print('score =', get_accuracy(Ytest, Ypred))
		score+=get_accuracy(ytrain[test_index], ypredict)		
		
	score/=k_fold
	print('score=',score)
	print()
	print(i)
	

        
#Function to open test set and generate predicted output    
def get_output(estimator,write,name):
    data = pandas.read_hdf('test.h5', "test")
    index=[]
    for key in data.keys():
        if(key!="Id"):
            index.append(key)
    X_test=np.array(data[index])
    #poly=skpr.PolynomialFeatures(degree)
    #X_test_final=poly.fit_transform(X_test)
    X_test_final = np.array(X_test, dtype = 'float128')
    Y_test=estimator.predict(X_test_final)
    print(Y_test)
    
    if(write):
        write_output_file(Y_test,name)


######################### MAIN #############################
if __name__ == '__main__':

	train = pandas.read_hdf("train.h5", "train")
	test = pandas.read_hdf("test.h5", "test")
	

	#create array x_train for training data
	index=[]
	for key in train.keys():	
		if(key!="Id" and key!="y"):
		    index.append(key)        
	x_train=np.array(train[index])

	#create array y_train in training data
	y_train=np.array(train['y'])
	
	#create array x_test in test set
	index_test = []
	for key in test.keys():	
		if(key!="Id" and key!="y"):
		    index_test.append(key)        
	x_test=np.array(test[index_test])
	
	
	def zero():
		return np.array([[1, 0, 0, 0, 0]])

	def one():
		return np.array([[0, 1, 0, 0, 0]])

	def two():
		return np.array([[0, 0, 1, 0, 0]])
	
	def three():
		return np.array([[0, 0, 0, 1, 0]])

	def four():
		return np.array([[0, 0, 0, 0, 1]])

	# map the inputs to the function blocks
	options = {
		0 : zero,
    	1 : one,
    	2 : two,
    	3 : three,
    	4 : four,
	}
	
	#change y_train into one-hot code
	y_train_oneHot = np.array(options[y_train[0]]())
	for i in range(1, len(y_train)):		
		y_train_oneHot = np.append(y_train_oneHot,options[y_train[i]](), axis=0)
		
		
	#######start the tensorflow manipulation#########
	x = tf.placeholder(tf.float32, [None, 100])
	W = tf.Variable(tf.zeros([100, 5]))
	b = tf.Variable(tf.zeros([5]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	y_ = tf.placeholder(tf.float32, [None, 5])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	
	init = tf.initialize_all_variables()

	sess = tf.Session()
	sess.run(init)
	
	
	len_data = len(x_train)

	for i in range(1000):
		index2 = np.random.choice(len_data, 100)  # using stochastic batches of the training set to train the neural network
		xs_batch = np.array(x_train[index2])
		ys_batch = np.array(y_train_oneHot[index2])
		sess.run(train_step, feed_dict={x: xs_batch, y_: ys_batch})

	#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	y_predict = sess.run(tf.argmax(y, 1), feed_dict={x: x_test})
	#print classification
	
	write_output_file(y_predict,"project3.csv")
	#get_output(mlp,true,"project3.csv")
'''
	#Split data into training and test (change train_size)
	#Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.9)

	
	crossValidation(clf,X,Y,k_fold=10)  #  do cross validation on k-folds and generate a validation score
	
	Y_predict = clf.predict(X)
	output_to_file=True
	get_output(clf,output_to_file,"project3.csv")
'''

