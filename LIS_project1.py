import csv
import numpy as np
import matplotlib.pylab as plt

x_train=[]
y_train=[]

first_line=True
q=1
with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n')
    for row in reader:
        if(first_line):
            first_line=False
            print(row)
            continue
        else: 
            #if(q<3):
            entry=row[0].split(',')
            x_rows=entry[2::]
            for i in range(len(x_rows)):
                x_rows[i]=float(x_rows[i])  
            x_train=np.hstack((x_train,entry))
            y_train=np.hstack((y_train,entry[1]))
            #print(entry)
            #print(x_train)
            #    q=q+1
                
    print("FINAL:")
    #print(x_train)
    print(int(len(x_train)/15))
    x_train=np.reshape(x_train,(int(len(x_train)/15),15))
    print(x_train)
    print(y_train)
    print(len(y_train))
