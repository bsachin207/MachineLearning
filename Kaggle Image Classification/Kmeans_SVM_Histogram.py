'''
Name: Sachin Badgujar
Hotel Image classification project part-1

Code Description-
This code extract top 500 SIFT descriptor are extracted from an image
Features from all training images is stored in csv file. This file is used in
next part for K-Means clustering.

Main purpose:
To save memory and cpu computations every time we make a program run
To focus on machine learning more.
'''

import csv
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
from sklearn.svm import SVC
from numpy import genfromtxt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

def reformatY(Y):
    new_Y = {}
    Y1 = Y.tolist()
    for i in range(1,len(Y)):
        label =int(Y1[i].index(1))
        new_Y[int(Y1[i][0])]=label
    return new_Y

        
train_Y = genfromtxt('C:/Users/sbadguja/Downloads/project/train.csv', delimiter=',')
SIFT_data = genfromtxt('C:\ExtractSIFT.csv', delimiter=',',max_rows=900000)
train_folder = 'C:/Users/sbadguja/Downloads/project/Training/'
test_folder = 'C:/Users/sbadguja/Downloads/project/Testing/'

#Create Label Dictionary
dict_Y = reformatY(train_Y)
#Initializations
train_img_ids = []
test_img_ids = []
f_vectors = []
train_f_vectors = []
test_f_vectors = []
train_f_hist = []
test_f_hist = []
images=[]

#Found elbow at 400-500
N_cluster = 500
#These values are based on various validations and details are mentioned in the Report
SIFT_WEIGHT = 0.90                                          
HIST_WEIGHT = 0.10

#Accuracy and Performance trade off is found out with MiniBatchKmeans algorithm
ms = MiniBatchKMeans(n_clusters=N_cluster,max_no_improvement=3,batch_size=20000)
ms.fit(SIFT_data)
sumimages = 0
del SIFT_data

#*************************************Read Every training Image*********************************
list1 = os.listdir(train_folder)
for filename in list1:
    try:
            img = cv2.imread(train_folder+filename,0)
            sumimages +=1
            if img is not None:
                sft = cv2.SIFT(300)
                kp,ds=sft.detectAndCompute(img,None)
                if len(ds) > 0:
                    cluster_predicted = ms.predict(ds)
                    SIFT_hist = np.bincount(cluster_predicted,minlength=N_cluster)
                    gray_hist = cv2.calcHist(img,[0],None,[64],[0,256])
                    train_img_ids.append(filename.split('.')[0])
                    train_f_vectors.append(SIFT_hist)
                    train_f_hist.append(gray_hist)
                else:
                    images.append(filename.split('.')[0])
            else:
                 images.append(filename.split('.')[0])
    except:
        images.append(filename.split('.')[0])
        pass
    if sumimages%1000==0:
            print sumimages


images=[]
train_f_vectors_hist = np.squeeze(train_f_hist,axis=(2,))
Y = [dict_Y[int(x)] for x in train_img_ids]

#Train SIFT features
clf_SIFT = SVC(kernel='rbf', cache_size=4096,probability=True)
clf_SIFT.fit(train_f_vectors,Y)  

#Train Histogram features
clf_HIST = KNeighborsClassifier(n_neighbors=50,n_jobs=-1)
clf_HIST.fit(train_f_vectors_hist,Y)   

del train_f_hist
del train_f_vectors
del train_f_vectors_hist
train_f_vectors = []

#*************************************Read Every Testing Image*********************************
sumimages = 0
list1 = os.listdir(test_folder)
for filename in list1:
    try:
            img = cv2.imread(test_folder+filename,0)
            sumimages +=1
            if img is not None:
                sft = cv2.SIFT(300)
                kp,ds=sft.detectAndCompute(img,None)
                if len(ds) > 0:
                    cluster_predicted = ms.predict(ds)
                    SIFT_hist = np.bincount(cluster_predicted,minlength=N_cluster)
                    gray_hist = cv2.calcHist(img,[0],None,[64],[0,256])
                    test_img_ids.append(filename.split('.')[0])
                    test_f_vectors.append(SIFT_hist)
                    test_f_hist.append(gray_hist)
                else:
                    images.append(filename.split('.')[0])
            else:
                 images.append(filename.split('.')[0])
    except:
        images.append(filename.split('.')[0])
        pass
    if sumimages%1000==0:
        print sumimages            

print 'Predicting Now'
test_f_vectors_hist = np.squeeze(test_f_hist,axis=(2,))
Y_predicted_SIFT = clf_SIFT.predict_proba(test_f_vectors)
Y_predicted_HIST = clf_HIST.predict_proba(test_f_vectors_hist)

#Take votes of SIFT predictions and Histogram prediction. Weight calculated from validation accuracy 
weighted_prediction = Y_predicted_SIFT * SIFT_WEIGHT + Y_predicted_HIST * HIST_WEIGHT

#Write CSV
header_row = ('id','col1','col2','col3','col4',	'col5',	'col6',	'col7',	'col8')
temp_csv = np.insert(weighted_prediction,0,test_img_ids,axis=1)
temp_csv_list = temp_csv.tolist()
temp_csv_list.insert(0,header_row)
write_csv = np.insert(temp_csv_list,0,header_row,axis=0)
with open('ImagePredictions.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(temp_csv_list)
    

#Automatically assigns equal probability to the corrupted images.
if len(images)>1:
        
    missing_values = np.array([[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]]*len(images))
    temp_csv = np.insert(missing_values,0,images,axis=1)
    temp_csv_list = temp_csv.tolist()
    
    with open('ImagePredictions.csv', "a") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(temp_csv_list)
else:
    print images
