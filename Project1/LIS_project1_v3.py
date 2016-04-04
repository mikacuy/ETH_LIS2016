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
features=[]
x13=[]
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
    x_test_sq=x_test**2
    x_test_cube=x_test**3
    x_test_final=np.hstack((x_test,x_test_sq))
    x_test_final=np.hstack((x_test_final,x_test_cube))
    y_test=clf.predict(x_test_final)
    print("TEST SET Y values: with "+str(len(y_test))+" samples")
    print(y_test)
    
    if(write):
        write_output_file(y_test,name)

def get_output_x13(xxx,write,name):
    line=True
    x13_test=[]
    with open('test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            if(line):
                line=False
                continue
            else: 
                x13_row=[]
                entry=row[0].split(',')
                x_rows=entry[1::]
                for i in range(len(x_rows)):
                    if(i==12):
                        x13_row.append(float(x_rows[i]))
                        x13_test.append(x13_row)

    x13_test = np.array(x13_test, dtype = 'float_')
    y_test=xxx.predict(x13_test)
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
            entry=row[0].split(',')
            for i in range(len(entry)-2):
                features.append(entry[i+2])
            print("features",features)
            continue
        else: 
            entry=row[0].split(',')
            x_rows=entry[2::]
            x13_row=[]
            for i in range(len(x_rows)):
                x_rows[i]=float(x_rows[i])
                if(i==12):
                    #x13_row.append(1.0)
                    x13_row.append(float(x_rows[i]))
                    x13.append(x13_row)
            x_train=np.hstack((x_train,x_rows))
            y_train=np.hstack((y_train,entry[i]))

          
x_train=np.reshape(x_train,(int(len(x_train)/15),15))
x_train = np.array(x_train, dtype = 'float_')
y_train = np.array(y_train, dtype = 'float_')

'''
print()
print("Find correlations")
XYtrain = np.vstack((x_train.T, np.atleast_2d(y_train)))
correlations = np.corrcoef(XYtrain)[-1, :]
print('features', len(features), 'correlations', len(correlations))
for feature_name, correlation in zip(features, correlations):
    print('{0:>10} {1:+.4f}'.format(feature_name, correlation))
print('{0:>10} {1:+.4f}'.format('OUTPUT', correlations[-1]))
'''

#fit x13
print(x13)
#print("len x13",len(x13))
#print("len y",len(y_train))    
X13train, X13test, Y13train, Y13test = skcv.train_test_split(x13, y_train, train_size=0.9)
#print('Shape of Xtrain:', Xtrain.shape)
#print('Shape of Ytrain:', Ytrain.shape)
#print('Shape of Xtest:', Xtest.shape)
#print('Shape of Ytest:', Ytest.shape)
regressor=sklin.LinearRegression()
#regressor = sklin.LassoLars()
#clf=sklin.Ridge(alpha=0.0)
#regressor=sklin.BayesianRidge(compute_score=True)
#clf=sklin.TheilSenRegressor(random_state=42)
#clf=sklin.RANSACRegressor(random_state=42)
regressor.fit(X13train,Y13train)
Y13pred = regressor.predict(X13test)
print('x13 score =', rms(Y13test, Y13pred))
#print('x13 score2 =', regressor.score(X13test,Y13pred))
print()
#plt.plot(x13,y_train,'ro')
#plt.plot(X13train,Y13train,'ro')
#plt.plot(X13test,Y13test,'b')
#get_output_x13(regressor,True,"x13.csv")

degree=1
poly=skpr.PolynomialFeatures(degree)
x_final=poly.fit_transform(x13)

Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(x_final, y_train, train_size=0.5)
clf = sklin.LinearRegression()
clf.fit(Xtrain,Ytrain)
Ypred = clf.predict(Xtest)
print('x13 deg',degree,' score =', rms(Ytest, Ypred))
#get_output_x13(regressor,True,"x13_.csv")
x = np.linspace(0, 10, 100)
#plt.plot(x,clf.predict(x),'b-')
#plt.plot(Xtrain,Ytrain,'ro')
#plt.plot(Xtest,Ytest,'bo')
#plt.plot(Xtest,Ypred,'g-')

regressor_ridge = sklin.Ridge()
param_grid = {'alpha': np.linspace(0, 100, 10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -rms(x, y))  # Note the negative sign.
grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring='r2', cv=5)
grid_search.fit(Xtrain, Ytrain)
best = grid_search.best_estimator_
print()
print(best)
print('best score =', -grid_search.best_score_)


#get_output_x13(grid_search.best_estimator_,True,"x13_2.csv")


'''
poly=skpr.PolynomialFeatures(degree=2)
x_final=poly.fit_transform(x_train)
print(x_final)
print(len(x_final[0]))
print(len(x_train[0]))
print()

#Split the data
Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(x_final, y_train, train_size=0.9)
#print('Shape of Xtrain:', Xtrain.shape)
#print('Shape of Ytrain:', Ytrain.shape)
#print('Shape of Xtest:', Xtest.shape)
#print('Shape of Ytest:', Ytest.shape)

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
print('score2 =', clf.score(Xtest,Ypred))

print()
scorefun = skmet.make_scorer(rms)
scores = skcv.cross_val_score(clf, x_final, y_train, scoring=scorefun, cv=5)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))

#get_output(clf,True,"theil_sen.csv")

regressor_ridge = sklin.Ridge()
param_grid = {'alpha': np.linspace(0, 100, 10)}
neg_scorefun = skmet.make_scorer(lambda x, y: -rms(x, y))  # Note the negative sign.
grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring=neg_scorefun, cv=5)
grid_search.fit(Xtrain, Ytrain)

best = grid_search.best_estimator_
print()
print(best)
print('best score =', -grid_search.best_score_)
'''
