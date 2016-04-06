# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:27:54 2016

@author: mikacuy
"""

import csv
import numpy as np
import pandas

import sklearn.cross_validation as skcv
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
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.9)


#svm with linear/polynomial/gaussian kernel
#for kernel in ('linear', 'poly', 'rbf'):
clf = SVC(kernel='poly',gamma=2)
clf.fit(Xtrain,Ytrain)
Ypred=clf.predict(Xtest)
print('score =', get_accuracy(Ytest, Ypred))
print()
#output_to_file=True
#get_output(clf,output_to_file,"project2_svm_poly.csv")


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

