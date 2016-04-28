# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 18:22:52 2016
Machine learning homework 3
@author: Sachin Badgujar
"""
import os
import numpy as np
import csv
from itertools import islice
import random
import math

class ML:
    def __init__(self):
        self.alpha =[]
        self.hypothesis=[]
        self.algo = ""
    def setAlpha(self,a):
        for i in a:
            self.alpha.append(i)
    def setHypothesis(self,h):
        for i in h:
            self.hypothesis.append(i)
    def setAlgo(self,a):
        self.algo = a
              
        
    

def adaTrain(XTrain,YTrain,version):
    
    YValidation = YTrain[:int(cnt*0.25)]
    XValidation = XTrain[:int(cnt*0.25)][:]
    
    if (version=="perceptron"):
        Y = YTrain[int(cnt*0.25)+1:]
        X = XTrain[int(cnt*0.25)+1:][:]
        model=perceptron(X,Y,XValidation,YValidation)
    else:
        model=stumps(XTrain,YTrain,XValidation,YValidation)
    return model
    
def perceptron(XTrain,YTrain,XValidation,YValidation):
    #initial weights of 1/m
    X_new = np.insert(XTrain,0,1,axis=1)
    XValidation = np.insert(XValidation,0,1,axis=1)
    X_data_weights = np.array([1/float(len(XTrain)) for x in range(len(XTrain))])    #data Weights
    alpha = []
    pre_validation_err = 9999
    validation_err= 99999
    k= 0
    w = 0
    min_iter = 0
    h=np.zeros(17)

    for k in range(20):
    #while(k<20 and pre_validation_err>validation_err):
        
        w = pseudoinverse(X_new, YTrain)
        [w, min_iter] = pla(X_new,YTrain,w,X_data_weights)
        verify = perceptrontest(XValidation,YValidation,w)
        miss = np.where(verify == -1)[0]
        pre_validation_err = validation_err
        
        validation_err = len(miss)/float(len(YValidation))
        #break the loop when its overfitting
        if (pre_validation_err<validation_err):
            break;
     
        if k == 0:
            h=w       
        if k > 0 :
            h = np.vstack([h,w])  
        
        #Find training error        
        check = perceptrontest(X_new,YTrain,w)
        misclassified = np.where(check == -1)[0]
        misclassified_weigts = [X_data_weights[j] for j in misclassified]
        epsi = sum(misclassified_weigts)
        alpha.append(math.log((1-epsi)/epsi)/2)
        X_data_weights_prev = np.asarray(X_data_weights) 
        #update weights
        for j in range(len(X_data_weights)):
            X_data_weights[j] = X_data_weights_prev[j]*math.exp(-alpha[k]*np.sign(check)[j])
        X_data_weights = X_data_weights/float(sum(X_data_weights))
        print "Iteration ",k,"-" ,validation_err
        k=k+1


    model.setAlgo("perceptron")
    model.setAlpha(alpha)
    model.setHypothesis(h)
    return model


def perceptrontest(X,Y,w):
    result  = np.dot(w,X.T) 
    check = result*Y
    return np.sign(check)
    
def pla(X_new,Y,w,D_w):
  
    error = []
    pocket_weights = []
    
    #200 loop if data is not linearly seperable.
    for i in range(0,200):
        comp = perceptrontest(X_new,Y,w) <= 0                      #Compare will be array of booleans- True for misclassified point
        if((~comp.any())):
            break                                                   #If no-misclassified point break the loop

        #pick the misclassified point with highest weight
        misclassified = np.where(comp == True)[0]   
        #print misclassified
        misclassified_weigts = [D_w[j] for j in misclassified]
        indx = misclassified[misclassified_weigts.index(max(misclassified_weigts))]        
        #maintain pocket errors and weights        
        error.append(float(len(misclassified)) / len(X_new))
        w = w + Y[indx]*X_new[indx]            # Calculating new weights
        if i == 0:
            pocket_weights = w        
        if i > 0 :
            pocket_weights = np.vstack([pocket_weights,w])        
        
    #picking weights from pocket - min error
    w_final = pocket_weights[error.index(min(error))-1] 
    return w_final,error.index(min(error))
    
#Pseudo invesrse for intial weights of perceptron   
def pseudoinverse(X_new, Y):
    
    w = np.linalg.pinv(X_new)
    return w.dot(Y)

#Evaluates the hypothesis
def stumptest(X,h):
    Yout = np.zeros(len(X))
    for j in range(len(X)):
        if X[j][h[0]] == 1:
            Yout[j] = h[1]
        else:
            Yout[j] = h[2]
    return Yout

#Implementation of Decision stumps       
def stumps(XTrain,YTrain,Xvalidation,YValidation):
    
    h=[]
    alpha = []
    validation_err = 9999
    smodel = ML()
    smodel_temp = ML()
    
    #initial weights of 1/m
    X_data_weights = np.array([1/float(len(XTrain)) for x in range(len(XTrain))])
    
    for k in range(20):
        h.insert(k,buildstump(XTrain,YTrain,X_data_weights,h))
        YCalculated = stumptest(XTrain,h[k])
        result = YCalculated*YTrain
        misclassified  = np.where(result==-1)
        #calcularte in sample error
        if len(misclassified[0]) == 0:
            epsi = 0            
            break;
        else:
            w_sum = 0
            for t in misclassified[0]:
                w_sum = w_sum + X_data_weights[t]
                epsi = w_sum
                
        alpha.insert(k,math.log((1-epsi)/epsi)/2)
        #Maintain a model for validation error        
        smodel_temp.setAlgo("stumps")
        smodel_temp.setAlpha(alpha)
        smodel_temp.setHypothesis(h)
        #find validation error        
        Yout = adaPredict(smodel_temp,Xvalidation)
        verify = Yout*YValidation
        prev_validation_err = validation_err
        validation_err = (len(np.where(verify==-1)[0]))/float(len(YValidation))
        
        #if overfittin        
        if (prev_validation_err < validation_err):
            break
        
        smodel.setAlgo("stumps")
        smodel.setAlpha(alpha)
        smodel.setHypothesis(h)        

        #update data weights        
        X_data_weights_prev = np.asarray(X_data_weights) 
        for j in range(len(X_data_weights)):
            X_data_weights[j] = X_data_weights_prev[j]*math.exp(-alpha[k]*np.sign(result)[j])
        X_data_weights = X_data_weights/float(sum(X_data_weights))    
    
    print "Iteration-", k
    return smodel

#Assumption: if 0 check majority of the answer and assign. if equal votes, pick a random sign    
def buildstump(X,Y,X_data_weights,h_done):
    row, col = X.shape
    h = []
    error = []
    for i in range(col):
        p_sum_posi = 0
        p_sum_neg = 0
        n_sum_neg = 0
        n_sum_posi = 0
        positives = np.where(X[:,i]==1)
        for j in positives[0]:
            if Y[j]==1:
                p_sum_posi = p_sum_posi + X_data_weights[j]
            if Y[j]==-1:
                p_sum_neg = p_sum_neg + X_data_weights[j]
        
        negatives = np.where(X[:,i]==-1)
        for j in negatives[0]:
            if Y[j]==1:
                n_sum_posi = n_sum_posi + X_data_weights[j]
            if Y[j]==-1:
                n_sum_neg = n_sum_neg + X_data_weights[j]
            
        
        label_when_1 = np.sign(p_sum_posi - p_sum_neg)
        label_when_m1 = np.sign(n_sum_posi - n_sum_neg) 
        
        if label_when_1 == 1 and label_when_m1 == 1:
            total_err = p_sum_neg + n_sum_neg
        if label_when_1 == 1 and label_when_m1 == -1:
            total_err = p_sum_neg + n_sum_posi
        if label_when_1 == -1 and label_when_m1 == 1:
            total_err = p_sum_posi + n_sum_neg
        if label_when_1 == -1 and label_when_m1 == -1:
            total_err = p_sum_posi + n_sum_posi
            
        error.append(total_err)
    
        h.append([i,label_when_1,label_when_m1])
    
    if len(h_done) != 0:
        for d in h_done:
            error[d[0]] = 9999
    
    return h[error.index(min(error))]
        

def fixdata(X):
    for i in range(len(X)):
        zero_array = np.where(X[i]==0)
        for j in range(len(zero_array)):
            col = zero_array[j]
            replace = np.sign(np.sum(X[:][col]))
            if replace==0:
                X[i][col]=random.choice([1,-1])
            else:
                X[i][col] = replace
    return X
 
def adaPredict(model,XTest):
     if model.algo == "perceptron":
         temp = []
         total = [[]]
         i=0
         Y = np.zeros(len(XTest))
         if len(model.alpha) == 1:
             result = np.sign(np.dot(model.hypothesis,XTest.T))
             temp = np.array(model.alpha[i]*result)
         else:
             for i in range(len(model.alpha)):
                 result = np.sign(np.dot(model.hypothesis[i],XTest.T))
                 temp.insert(i,(model.alpha[i]*result))
         
         total = np.array(temp)
 
         #caluclate Y
         if(len(model.alpha)==1):
             Y = np.sign(total)
         else:
             for j in range (len(XTest)):
                 
                 Y[j] =  np.sign(sum(total[:,j]))
     else:
        result = []
        temp=[]
        total = []
        Y=np.zeros(len(XTest))
        for i in range(len(model.alpha)):
            #print "hypo- ",model.hypothesis[i]
            result = np.sign(stumptest(XTest,model.hypothesis[i]))
            temp.append(result*model.alpha[i])
        
        total  = np.array(temp)
        
        for j in range (len(XTest)):
                Y[j] =  np.sign(sum(total[:,j]))

     return Y
     
if __name__ == "__main__":
    fo = open("D:\house-votes-84.data")
    #random.shuffle(fo)
    datalist = []
    reader = (csv.reader(fo))
    cnt = 0
    
    for row in reader:
        datalist.append([x.replace('democrat','-1').replace('republican','1').replace('y','1').replace('n','-1').replace('?','0') for x in row])
        cnt=cnt+1
        
    accuracy = []
    for n in range(10):
        random.shuffle(datalist)
        y_data = np.asarray([datalist[i][0] for i in range(len(datalist))]).astype(np.int)
        x_data = np.asarray([datalist[i][1:] for i in range(len(datalist))]).astype(np.int)
        
        x_data_fixed = fixdata(x_data)
        YTrain = y_data[:int(cnt*0.75)]
        XTrain = x_data_fixed[:int(cnt*0.75)][:]
        
        YTest_True = y_data[int(cnt*0.75):]
        XTesting = x_data_fixed[int(cnt*0.75):][:]
        XTest = XTesting    
        
        #Uncomment the below lines as per requirement
        version = "stumps"
        #version = "perceptron"
        if version == "perceptron":
            XTest = np.insert(XTesting,0,1,axis=1)
 
        model = ML()
        model = adaTrain(XTrain,YTrain,version)
        
        YTest = adaPredict(model,XTest)
        #print model.hypothesis
        
        cr_validation = YTest_True*YTest
        #print np.where(cr_validation == -1)
        accuracy.insert(n,(1-(sum(cr_validation==-1)/float(len(cr_validation))))*100)
        
    print "Accuracy- ", accuracy
    print "Min Accuracy-", min(accuracy)
    print "Average Accuracy-", np.average(accuracy)
    print "Max Accuracy-", max(accuracy)
    

         
         
    
    

    
