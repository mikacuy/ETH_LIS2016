# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:06:48 2016

@author: mikacuy
"""

import csv
import numpy as np
import matplotlib.pylab as plt

# Contains linear models, e.g., linear regression, ridge regression, LASSO, etc.
import sklearn.linear_model as sklin
# Allows us to create custom scoring functions
import sklearn.metrics as skmet
# Provides train-test split, cross-validation, etc.
import sklearn.cross_validation as skcv
# Provides grid search functionality
import sklearn.grid_search as skgs
# The dataset we will use
from sklearn.datasets import load_boston
# For data normalization
import sklearn.preprocessing as skpr


x_train=[]
y_train=[]
first_line=True

def rms(gtruth, pred):
    diff = gtruth - pred
    return np.sqrt(np.mean(np.square(diff)))
    
def write_output_file(y_test,name):
    #Write to an output file
    with open(name, 'wt') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Id','y'))
        line=[0,0.0]
        for i in range(len(y_test)):
            line[0]=900+i
            line[1]=y_test[i]
            #print(line)
            writer.writerow(line)

#Open test data
def get_output(clf,write,name):
    line=True
    x_test=[]
    with open('test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            if(line):
                line=False
                continue
            else: 
                entry=row[0].split(',')
                x_rows=entry[1::]
                for i in range(len(x_rows)):
                    x_rows[i]=float(x_rows[i])  
                x_test=np.hstack((x_test,x_rows))

    x_test=np.reshape(x_test,(int(len(x_test)/15),15))
    y_test=clf.predict(x_test)
    print("TEST SET Y values: with "+str(len(y_test))+" samples")
    print(y_test)
    
    if(write):
        write_output_file(y_test,name)
    

#Open training data
with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n')
    for row in reader:
        if(first_line):
            first_line=False
            continue
        else: 
            entry=row[0].split(',')
            x_rows=entry[2::]
            for i in range(len(x_rows)):
                x_rows[i]=float(x_rows[i])  
            x_train=np.hstack((x_train,x_rows))
            y_train=np.hstack((y_train,entry[i]))
            
x_train=np.reshape(x_train,(int(len(x_train)/15),15))
x_train = np.array(x_train, dtype = 'float_')
y_train = np.array(y_train, dtype = 'float_')

#Split the data
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(x_train, y_train, train_size=0.8)
print('Shape of Xtrain:', Xtrain.shape)
print('Shape of Ytrain:', Ytrain.shape)
print('Shape of Xtest:', Xtest.shape)
print('Shape of Ytest:', Ytest.shape)

#Models
clf = sklin.LinearRegression()
#clf = sklin.Lasso()
#clf=sklin.Ridge(alpha=0.0)
#clf=sklin.BayesianRidge(compute_score=True)
#clf=sklin.TheilSenRegressor(random_state=42)
#clf=sklin.RANSACRegressor(random_state=42)
clf.fit(Xtrain,Ytrain)
Ypred = clf.predict(Xtest)
print('score =', rms(Ytest, Ypred))

scorefun = skmet.make_scorer(rms)
scores = skcv.cross_val_score(clf, x_train, y_train, scoring=scorefun, cv=5)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))

get_output(clf,True,"test_output.csv")



