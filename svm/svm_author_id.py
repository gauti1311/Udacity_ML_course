#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels   
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC(kernel='rbf', C = 10000)
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
t0 = time()
clf.fit(features_train,labels_train)
t1 = time()
pred = clf.predict(features_test)
t2 = time()
acc = accuracy_score(pred,labels_test)
print("accuracy of prediction is :%.3f" %acc)
count_chris = 0
count_sara = 0

for i in range(len(pred)):
    if pred[i] == 1:
        count_chris=count_chris+1
    else: 
        count_sara=count_sara+1
         
print ("no. of prediction of Chris: %d and Sara:%d" %(count_chris,count_sara))

  

#########################################################


