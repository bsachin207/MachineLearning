
'''
Name: KNN-Algorithm and Condensed Algorithm
@author: Sachin Badgujar (sbadguja@uncc.edu)

'''

import random
import numpy as np
from itertools import islice
from sklearn.metrics import confusion_matrix
import time

#finding out euclidian distance with array
def myeuclidian(x,y):
    return np.sqrt(np.sum((x-y)**2))

#Calculate euclidian distance for condendsing alforithm
def condense_euclidian_distance(trainX, testX):
    dist = np.zeros((len(testX),1))
    for i in range(len(testX)):
        dist[i] = myeuclidian(testX[i],trainX)
    return dist
   
#Euclidian distance
def euclidian_distance(trn_X,tst_X):
    dist = np.zeros((len(tst_X),len(trn_X)))
    for i in range(len(tst_X)):
        for j in range(len(trn_X)):
            dist[i][j]=(myeuclidian(trn_X[j],tst_X[i]))
          #dist[i][j]=(distance.euclidean(trn_X[j],tst_X[i]))
    return dist

#K-NN algorithm
def testknn(trainX,trainY,testX,k):
    calculated_result = euclidian_distance(trainX,testX)
    knn_output = []
    for i in range(len(calculated_result)):
        ids = calculated_result[i].argsort()[:k].tolist()
        min_labels = [trainY[elm] for elm in ids]       
        knn_output.append(max(set(min_labels),key=min_labels.count))
    return knn_output

#Calculating accuracy of knn
def find_accuracy(desired,obtained):
    count = 0
    for i in range(len(desired)):
        if desired[i] == obtained[i]:
            count=count+1
    return (float(count)/float(len(desired)))*100

#Another version of verifying the results like find_accuracy
def check_classification(desired,obtained):
    unclassified = []
    for i in range(len(desired)):
        if desired[i] != obtained[i]:
            unclassified.append(i)
    return unclassified

#Function to find out condense set which is boundary consistant
def condensedata(trainX,trainY):
    indx = []
    knn_output = []
    not_classified = []
    not_classified.append(0)
    indx.append(not_classified[0])
    dst = condense_euclidian_distance(trainX[indx[0]], trainX)
    distance_matrix = np.array(dst)
    
    for i in range(len(dst)):
        label_index = distance_matrix[i].argmin()
        knn_output.append(trainY[indx[label_index]])
    not_classified = check_classification(trainY,knn_output)    
    
    knn_output = []
    while(not_classified and len(indx)<len(trainX)):
        indx.append(random.choice(not_classified))
        dst = condense_euclidian_distance(trainX[indx[-1]], trainX)
        distance_matrix = np.c_[distance_matrix,dst]

        for i in range(len(distance_matrix)):
            label_index = distance_matrix[i].argmin()
            knn_output.append(trainY[indx[label_index]])

        not_classified = check_classification(trainY,knn_output)
        knn_output = []
    return indx
    

	
#The main function

fo = open("D:\Machine Learning\letter-recognition.data").readlines()
random.shuffle(fo)
#Splitting data into training and testing
training_data = islice(fo,0,100)
testing_data = islice(fo,15000 ,20000)
#value of k
k=1
train_dataY = []
train_dataX = []
test_dataY = [] 
test_dataX = []

for lines in training_data:
    train_dataY.append(lines.strip().split(',')[0])
    train_dataX.append(map(int,(lines.strip().split(',')[1:])))
    
    
for lines in testing_data:
    test_dataY.append(lines.strip().split(',')[0])
    test_dataX.append(map(int,(lines.strip().split(',')[1:])))

#Converting it to numpy array.
trainX = np.array(train_dataX)
trainY = np.array(train_dataY)
testX = np.array(test_dataX)
result = np.array(test_dataY)

start_time = time.time()

#Prepare condense set
condensedIdx = condensedata(trainX, trainY)
print 'CondenseSet :',len(condensedIdx)
cond_trainY = [trainY[elm] for elm in condensedIdx]
cond_trainX = [trainX[elm] for elm in condensedIdx]

#run K-NN using new training data from Condense set
testY = testknn(cond_trainX,cond_trainY,testX,k)
accuracy = find_accuracy(result,testY)
print 'Accuracy :',accuracy

#printing confusion matrix into text file.
f1 = open('./ConfusionMatrix.txt','w')
conf_mat = confusion_matrix(result,testY)
print >> f1 ,list(set(result))
for i in range(len(conf_mat)):
    print >>f1,list(conf_mat[i])
f1.close()

#calculate elapsed time.
elapsed_time = time.time() - start_time
print 'Time :',elapsed_time