Name: Sachin Badgujar
Email: sbadguja@uncc.edu
Hotel Image Classification.

Summary:
Pertaining to the project approach, I have implemented modified bag of features with voted prediction. 
Considering all processing limitation, I have coded wisely and split project into two separate code as below:

ExtractSIFTtoCSV.py ----> ExtractSIFT.csv ----> Kmeans_SVM.py / Kmeans_SVM_Histogram.py ---> ImagePredictions.csv
(Program 1) --->      (Intermediate File)  ---->  (Program 2 - Kaggle best selection)	 ---->	   (Output file generated)


All programs are written using libraries like sklearn, cv2 version 2.4.9, numpy, csv.
References are taken from open sources like internet Machine Learning course material, IEEE papers, stackoverflow, scikit documentations. 
Development is done in Anaconda Python 2.7 64-Bit using spyder IDE. 

Instruction to run the code (Inputs need to be given manually).
1. Run ExtractSIFTtoCSV.py--
	Input: Training images folder path
	Output: ExtractSIFT.csv  in same working directory (Caution- consumes space)
You can by pass this step and directly get precomputed CSV at https://drive.google.com/a/uncc.edu/file/d/0BxQ3gjBIyZGxT2xmbjNpaUFISE0/view?usp=sharing (UNCC login needed)


2. Run Kmeans_SVM.py or Kmeans_SVM_Histogram.py
   Needs cv2 version 2.4.9 installed on Anaconda python 2.7 
   Input: ExtractSIFT.csv, Training images, Testing Images
   Output: ImagePredictions.csv  -- *No need to edit anything. Everything is ready (You may need to do 'Save As' csv if uploading on Kaggle).

   
I have selected these two submissions on Kaggle and corresponding two codes are included.
No parameters in code are random. Every parameter has a purpose and used appropriately.
More details about code functions are in Project report. Detail comments in the code are for sake of readability and understanding. 
   
 
   