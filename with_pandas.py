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
    
def get_output(xxx,write,name,degree):
    data = pandas.read_csv('test.csv', sep=',', na_values=".")
    print("In output file:")
    #print(data)
    index=[]
    for key in data.keys():
        if(key!="Id"):
            index.append(key)
    X_test=np.array(data[index])
    poly=skpr.PolynomialFeatures(degree)
    X_test_final=poly.fit_transform(X_test)
    #print(X_test)
    Y_test=xxx.predict(X_test_final)
    print(Y_test)
    
    if(write):
        write_output_file(Y_test,name)


data = pandas.read_csv('train.csv', sep=',', na_values=".")
print(data.keys())

data_to_plot=pandas.DataFrame({'x13':data['x13'],'y':data['y']})
model=ols("y~x13", data_to_plot).fit()
print(model.summary())

index=[]
for key in data.keys():
    if(key!="Id" and key!="y"):
        index.append(key)
        
X=np.array(data[index])
#print(X)
Y=np.array(data['y'])
#print(Y)

XYtrain = np.vstack((X.T, np.atleast_2d(Y)))
correlations = np.corrcoef(XYtrain)[-1, :]
print('features', len(index), 'correlations', len(correlations))
for feature_name, correlation in zip(index, correlations):
    print('{0:>10} {1:+.4f}'.format(feature_name, correlation))
print('{0:>10} {1:+.4f}'.format('OUTPUT', correlations[-1]))

'''
#split data and test
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.75)

#regressor = sklin.LinearRegression()
#regressor = sklin.Lasso()
regressor=sklin.Ridge(alpha=1.0)
#regressor=sklin.BayesianRidge(compute_score=True)
#regressor=sklin.TheilSenRegressor(random_state=42)
#regressor=sklin.RANSACRegressor(random_state=42)
regressor.fit(Xtrain, Ytrain)
#print('{0:>10} {1:+.4f}'.format('intercept', regressor.intercept_))
#for feature_name, coef in zip(index, regressor.coef_):
#    print('{0:>10} {1:+.4f}'.format(feature_name, coef))

Ypred = regressor.predict(Xtest)
print('score =', rms(Ytest, Ypred))
#get_output(regressor,True,"new.csv")

scorefun = skmet.make_scorer(rms)
scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))
'''

#higher order
degree=3
poly=skpr.PolynomialFeatures(degree,include_bias=True)
x_final=poly.fit_transform(X)
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(x_final, Y, train_size=0.9)
#clf = sklin.LinearRegression()
clf=sklin.BayesianRidge(compute_score=True)
#clf=sklin.Ridge(alpha=222.22)
clf.fit(Xtrain,Ytrain)
Ypred = clf.predict(Xtest)
print('x13 deg',degree,' score =', rms(Ytest, Ypred))
print()
#get_output(clf,True,"degree3_v3.csv",degree)

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

