import csv
import numpy as np
import matplotlib.pylab as plt
from sklearn import linear_model

x_train=[]
y_train=[]

first_line=True
#q=1

#Open training data
with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n')
    for row in reader:
        if(first_line):
            first_line=False
            #print(row)
            continue
        else: 
            #if(q<3):
            entry=row[0].split(',')
            x_rows=entry[2::]
            #print(x_rows)
            for i in range(len(x_rows)):
                x_rows[i]=float(x_rows[i])  
            x_train=np.hstack((x_train,x_rows))
            y_train=np.hstack((y_train,entry[i]))
            #print(entry)
            #print(x_train)
            #    q=q+1
                
print("TRAINING:")
#print(int(len(x_train)/15))
x_train=np.reshape(x_train,(int(len(x_train)/15),15))
print("X (nxd) with "+str(len(x_train))+" samples: ")
print(len(x_train))
print()
print("Y with "+str(len(y_train))+" samples: ")
print(y_train)
print()

    
#Ordinary Least Squares
x_train = np.array(x_train, dtype = 'float_')
y_train = np.array(y_train, dtype = 'float_')
clf = linear_model.LinearRegression()
clf.fit(x_train,y_train)

w=clf.coef_
print("Coefficients of Regression: "+"length: "+str(len(w)))
print(w)
print()

#Open test data
first_line=True
x_test=[]
with open('test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n')
    for row in reader:
        if(first_line):
            first_line=False
            continue
        else: 
            #if(q<3):
            entry=row[0].split(',')
            x_rows=entry[1::]
            #print(x_rows)
            for i in range(len(x_rows)):
                x_rows[i]=float(x_rows[i])  
            x_test=np.hstack((x_test,x_rows))
            #print(entry)
            #print(x_train)
            #    q=q+1
x_test=np.reshape(x_test,(int(len(x_test)/15),15))
print("TEST SET (kxd) with "+str(len(x_test))+" samples: ")
print(x_test)
print()        
    
    
    
    
    
    
'''
#Find w*
w=np.array(np.zeros(15))
print(w)
'''

'''

x=n*d ; d = 15, xi=15
y=n*1, yi = 1
w=d*1

we wanna make R(w) = SIGMA(yi-Xw)^2 + Lambda*|w|^2
so we do gradient descent on R(w)

'''

