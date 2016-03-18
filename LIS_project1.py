# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:02:47 2016

@author: mikacuy
"""

import csv
import numpy as np
import pandas
from pandas.tools import plotting
from statsmodels.formula.api import ols
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


def rms(gtruth, pred):
    diff = gtruth - pred
    return np.sqrt(np.mean(np.square(diff)))

#Function to write to a file
def write_output_file(y_test,name):
    #Write to an output file
    with open(name, 'wt') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Id','y'))
        line=[0,0.0]
        for i in range(len(y_test)):
            line[0]=900+i
            line[1]=y_test[i]
            writer.writerow(line)
        print("File ",name," written")

#Function to open test set and generate predicted output    
def get_output(xxx,write,name,degree):
    data = pandas.read_csv('test.csv', sep=',', na_values=".")
    index=[]
    for key in data.keys():
        if(key!="Id"):
            index.append(key)
    X_test=np.array(data[index])
    poly=skpr.PolynomialFeatures(degree)
    X_test_final=poly.fit_transform(X_test)
    Y_test=xxx.predict(X_test_final)
    print(Y_test)
    
    if(write):
        write_output_file(Y_test,name)

#Open training file and store in pandas DataFrame
data = pandas.read_csv('train.csv', sep=',', na_values=".")
print(data.keys())

index=[]
for key in data.keys():
    if(key!="Id" and key!="y"):
        index.append(key)

#Construct matrix X of independent variables
X=np.array(data[index])

#Vector Y
Y=np.array(data['y'])

#Display correlation among independent and dependent variables
XYtrain = np.vstack((X.T, np.atleast_2d(Y)))
correlations = np.corrcoef(XYtrain)[-1, :]
print('features', len(index), 'correlations', len(correlations))
for feature_name, correlation in zip(index, correlations):
    print('{0:>10} {1:+.4f}'.format(feature_name, correlation))
print('{0:>10} {1:+.4f}'.format('OUTPUT', correlations[-1]))

#Variable to set the degree
degree=3

#Construct matrix up to degree
poly=skpr.PolynomialFeatures(degree,include_bias=True)
x_final=poly.fit_transform(X)

#Split data into training and test (change train_size)
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(x_final, Y, train_size=0.95)
#clf = sklin.LinearRegression()
#clf=sklin.BayesianRidge(compute_score=True)
clf=sklin.Ridge(alpha=222.22)
clf.fit(Xtrain,Ytrain)
Ypred = clf.predict(Xtest)
print('X deg',degree,' score =', rms(Ytest, Ypred))
print()

#Get test data result
output_to_file=False
get_output(clf,output_to_file,"degree3_v3.csv",degree)

#To compare performance of different Ridge alphas
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

