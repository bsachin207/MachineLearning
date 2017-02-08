'''
Name: Sachin Badgujar
Hotel Image classification project part-1

Code Descriptoin-
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


#Read Image features -  Top 500 SIFT descriptor are extracted from an image
def readImages(folder):
    img_ids = []
    f_vector = []
    images = []
    list1 = os.listdir(folder)
    sumimages=0
    for filename in list1:
        try:
            
            img = cv2.imread(folder+filename,0)
            sumimages +=1
            if img is not None:
                sft = cv2.SIFT(500)
                kp,ds=sft.detectAndCompute(img,None)

                if len(ds) > 50:
                    f_vector.append(ds)
                    img_ids.append(filename.split('.')[0])
                    
            else:
                 images.append(filename)
        except:
            images.append(filename)
            pass
        if sumimages%1000==0:
            print 'Training Images read-',sumimages
     
    return img_ids, f_vector

        
#Main Function
if __name__ == "__main__":
       
    train_folder = 'C:/Users/sbadguja/Downloads/project/testV/'
        
    #Read all images and extract features.
    img_ids, f_vectors = readImages(train_folder)
    with open('ExtractSIFT.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for i in f_vectors:
            writer.writerows(i)
    
