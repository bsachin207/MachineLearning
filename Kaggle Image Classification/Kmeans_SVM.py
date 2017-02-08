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
from numpy import genfromtxt
from sklearn.svm import SVC
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def reformatY(Y):
    new_Y = {}
    Y1 = Y.tolist()
    for i in range(1,len(Y)):
        #label = 'col'+str(Y1[i].index(1))
        label =int(Y1[i].index(1))
        new_Y[int(Y1[i][0])]=label
    
    return new_Y



#read Training Labels 
train_Y = genfromtxt('C:/Users/sbadguja/Downloads/project/train.csv', delimiter=',')
#Pick max_rows for creating dictionary/ clusters
SIFT_data = genfromtxt('C:\ExtractSIFT.csv', delimiter=',',max_rows=900000)
train_folder = 'C:/Users/sbadguja/Downloads/project/Training/'
test_folder = 'C:/Users/sbadguja/Downloads/project/Testing/'

#create dictionary of Y labels
dict_Y = reformatY(train_Y)
#initialization
train_img_ids = []
test_img_ids = []
f_vectors = []
train_f_vectors = []
test_f_vectors = []
images=[]

#Number of clusters for K-Means
N_cluster = 500


#Accuracy and Performance trade off is found out with MiniBatchKmeans algorithm
ms = MiniBatchKMeans(n_clusters=N_cluster,max_no_improvement=3,batch_size=10000)
ms.fit(SIFT_data)

#Remove from memory
del SIFT_data

print 'Done Clustering - Learning Now'

#Read training images
sumimages = 0
list1 = os.listdir(train_folder)
for filename in list1:
    try:
            img = cv2.imread(train_folder+filename,0)
            sumimages +=1
            if img is not None:
                sft = cv2.SIFT(300)
                kp,ds=sft.detectAndCompute(img,None)
                #print filename.split('.')[0]
                if len(ds) > 1:
                    cluster_predicted = ms.predict(ds)
                    SIFT_hist = np.bincount(cluster_predicted,minlength=N_cluster)
                    train_img_ids.append(filename.split('.')[0])
                    train_f_vectors.append(SIFT_hist)
                else:
                    images.append(filename.split('.')[0])
            else:
                 images.append(filename.split('.')[0])
    except:
        images.append(filename.split('.')[0])
        pass
    if sumimages%1000==0:
            print 'Training Images read-',sumimages




Y = [dict_Y[int(x)] for x in train_img_ids]
clf = SVC(kernel='rbf', cache_size=4096,probability=True)
clf.fit(train_f_vectors,Y,sample_weight=None)           

#Make some space by deleting used variables.
del train_f_vectors
train_f_vectors = []
images= []
sumimages = 0
print 'Reading Testing Data'

#Read Testing Image
list1 = os.listdir(test_folder)
for filename in list1:
    try:
            img = cv2.imread(test_folder+filename,0)
            sumimages +=1
           # print filename.split('.')[0]
            if img is not None:
                sft = cv2.SIFT(300)
                kp,ds=sft.detectAndCompute(img,None)
                if len(ds) > 1:
                    cluster_predicted = ms.predict(ds)
                    SIFT_hist = np.bincount(cluster_predicted,minlength=N_cluster)
                    test_img_ids.append(filename.split('.')[0])
                    test_f_vectors.append(SIFT_hist)
                else:
                    images.append(filename.split('.')[0])
            else:
                 images.append(filename.split('.')[0])
    except:
        images.append(filename.split('.')[0])
        pass
    if sumimages%1000==0:
        print sumimages            


#Predict the Y labels
print 'Predicting Data'
Y_predicted = clf.predict_proba(test_f_vectors)

#Intentionally kept comments as those represents validation accuracy of different experiments
#Comment above algorithm and Uncomment if you want to run 1 
#*************************Neural is here***************************************
#nn=Classifier(
#    layers=[
#        Layer("Maxout", units=100, pieces=2),
#        Layer("Softmax")],
#    learning_rate=0.001,
#    n_iter=25)
#nn.fit(f_vectors,np.array(Y))
#Y_predicted=nn.predict_proba(test_f_vectors)
#Y_actual = [dict_Y[int(x)] for x in test_img_ids]
#accuracy = nn.score(test_f_vectors,Y_actual,sample_weight=[1 for i in range(len(test_f_vectors))])
#print accuracy*100
#*********************************end******************************************

#*****************************Random Forest- 38%*******************************
#Learn the images
#Y = [dict_Y[int(x)] for x in img_ids]
#clf = ensemble.RandomForestClassifier(n_jobs=1,bootstrap=True,max_features='None',)
#clf.fit(f_vectors,Y)
#Y_predicted=clf.predict_proba(test_f_vectors)
#Y_actual = [dict_Y[int(x)] for x in test_img_ids]
#accuracy = clf.score(test_f_vectors,Y_actual,sample_weight=[1 for i in range(len(test_f_vectors))])
#print accuracy*100
#*********************************end******************************************

#*****************************Decision Tree- 28%*******************************
#clf = tree.DecisionTreeRegressor(splitter='best',max_leaf_nodes=32)
#clf.fit(f_vectors, Y)
#Y_predicted= clf.predict_proba(test_f_vectors)
#******************************************************************************

#**********************************Knn 40%*************************************
#one_nearest = KNeighborsClassifier(n_neighbors=100)
#one_nearest.fit(train_f_vectors,Y)
#Y_predicted = one_nearest.predict_proba(test_f_vectors)
#******************************************************************************

#**********************************Knn- Infesible Nu***************************
#clf = NuSVC(nu=0.9,probability=True)
#clf.fit(f_vectors,Y[:0],sample_weight=None)
#******************************************************************************

#Create CSV

print 'Writing CSV'

header_row = ('id','col1','col2','col3','col4',	'col5',	'col6',	'col7',	'col8')
temp_csv = np.insert(Y_predicted,0,test_img_ids,axis=1)
temp_csv_list = temp_csv.tolist()
temp_csv_list.insert(0,header_row)
write_csv = np.insert(temp_csv_list,0,header_row,axis=0)
with open('ImagePredictions.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(temp_csv_list)

#Automatically assigns equal probability to the corrupted images. 
if len(images) > 0:  
    missing_values = np.array([[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]]*len(images))
    temp_csv = np.insert(missing_values,0,images,axis=1)
    temp_csv_list = temp_csv.tolist()
    
    with open('ImagePredictions.csv', "a") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(temp_csv_list)
else:
    print images
