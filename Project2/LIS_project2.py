# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:27:54 2016

@author: mikacuy
"""

import csv
import numpy as np
import pandas

import sklearn.cross_validation as skcv
from sklearn.cross_validation import KFold
import sklearn.preprocessing as skpr
from sklearn import neighbors, datasets
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

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
            line[0]=1000+i
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
def get_output(xxx,write,name):
    data = pandas.read_csv('test2.csv', sep=',', na_values=".")
    index=[]
    for key in data.keys():
        if(key!="Id"):
            index.append(key)
    X_test=np.array(data[index])
    #poly=skpr.PolynomialFeatures(degree)
    #X_test_final=poly.fit_transform(X_test)
    X_test_final = np.array(X_test, dtype = 'float128')
    Y_test=xxx.predict(X_test_final)
    print(Y_test)
    
    if(write):
        write_output_file(Y_test,name)


######################### MAIN #############################
data = pandas.read_csv('train2.csv', sep=',', na_values=".")
print(data.keys())

#create array X for training data
index=[]
for key in data.keys():	
    if(key!="Id" and key!="y"):
        index.append(key)        
X=np.array(data[index])

#create array Y in training data
Y=np.array(data['y'])

#Split data into training and test (change train_size)
#Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.9)


#####################################################################
############# Change the estimator here #############################
#####################################################################

#clf = neighbors.KNeighborsClassifier(5, 'uniform')	
clf = SVC(kernel='rbf',degree = 2,gamma=0.002) # 0.002 is so far the best gamma for the gaussian kernel

#####################################################################
#####################################################################

crossValidation(clf,X,Y,k_fold=10)  #  do cross validation on k-folds and generate a validation score
clf.fit(X,Y) #  do training on the entire set instead of 0.9 of the set
Y_predict = clf.predict(X)
output_to_file=True
get_output(clf,output_to_file,"project2_gaussian_gamma_0_002.csv")

'''
#svm with linear/polynomial/gaussian kernel
#for kernel in ('linear', 'poly', 'rbf'):
clf = SVC(kernel='poly',gamma=2)   # kernel can be 'rbf','poly' or 'linear'
clf.fit(Xtrain,Ytrain)
Ypred=clf.predict(Xtest)
print('score =', get_accuracy(Ytest, Ypred))
print()
#output_to_file=True
#get_output(clf,output_to_file,"project2_svm_poly.csv")
'''

'''
#try nearest neighbor approach

#################EDIT K HERE####################
n_neighbors = 5 #k
################################################

clf = neighbors.KNeighborsClassifier(n_neighbors, 'uniform')
clf.fit(X,Y)
Ypred=clf.predict(Xtest)
#print('score =', get_accuracy(Ytest, Ypred))
print()

output_to_file=True
get_output(clf,output_to_file,"project2_kNN_5.csv")
'''

'''
#try nearest neighbor approach

#################EDIT K HERE####################
n_neighbors = 10 #k
################################################

clf = neighbors.KNeighborsClassifier(n_neighbors, 'uniform')
clf.fit(Xtrain,Ytrain)
Ypred=clf.predict(Xtest)
print('score =', get_accuracy(Ytest, Ypred))
print()

#output_to_file=True
#get_output(clf,output_to_file,"project2_kNN_5.csv")
'''



'''
regressor_ridge = sklin.Ridge()
param_grid = {'alpha': np.linspace(0, 1000, 10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -rms(x, y))  # Note the negative sign.
grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain, Ytrain)

best = grid_search.best_estimator_
print(best)
print('best score =', -grid_search.best_score_)
#get_output(clf,True,"degree3.csv",degree)
'''

