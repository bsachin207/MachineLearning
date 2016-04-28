#Name: Sachin Badgujar
#Email: sbadguja@uncc.edu
#Subject: Machine learing homework 2

import random
import numpy as np
import matplotlib.pyplot as plt


def generateData(N):
    #Choosing a random points between -1 to 1.
    X_matrix  = [[random.uniform(-1,1) for x in range(2)] for x in range(N) ]
    line_points = [[random.uniform(-1,1) for x in range(2)] for x in range(2) ]
    line_points=np.array(line_points)
    #Drawing imaginary line from line Points.
    X_cordinates = line_points[1][0] - line_points[0][0]
    Y_cordinates = line_points[1][1] - line_points[0][1]
    
    #Checking wheather the point in X array lies in +1 side or -1 of the line
    Y_label = [((X_cordinates*(X_matrix[i][1] - line_points[0][1])) - (Y_cordinates*(X_matrix[i][0] - line_points[0][0])) > 0) for i in range(len(X_matrix))]
  
    X = np.array(X_matrix)
    Y = ((2*(np.array(Y_label)))-1)   			#Adjecement to covert 0,1 to -1,1, Y label is a boolean array
  
    '''    
    #Lines for plotting graphs. Uncomment if number of experiements are less.   
    color_array = np.array(['r','b','g'])
    plt.scatter(X[:,0],X[:,1],s=50,c=color_array[Y+1])
    plt.plot(line_points[:,0],line_points[:,1],marker='o',linestyle= '--',color='r')
    plt.show()
    '''
    #Returning the X and Y arrays to the main function    
    return X,Y
    
    
def pla(X,Y,w0):
    X_new = np.insert(X,0,1,axis=1)
    iteration  = 1
    
    #Infinite loop if data is not linearly seperable.
    while(1):
        result  = np.dot(w0,X_new.T)              #Calculate X.W
        comp = result*Y <= 0                      #Compare will be array of booleans- True for misclassified point
        if((~comp.any())):
            break                                 #If no-misclassified point break the loop

        misclassified = np.where(comp == True)[0]   
        indx = random.choice(misclassified)      # Pick a random misclassified example and get its index
        
        w0 = w0 + Y[indx]*X_new[indx]            # Calculating new weights
        iteration=iteration+1
        
    return w0,iteration
    
def pseudoinverse(X, Y):

    X_new = np.insert(X,0,1,axis=1)
    w = np.linalg.pinv(X_new)
    return w.dot(Y)
    
#Main function    
if __name__ == "__main__":
    
    #number of Data points -  Please change it manually    
    N = 100
    pla_iters = []
    pla_pseudo_iters = []
    for i in range(100):
        [X,Y] = generateData(N)
        w0 = np.array([0,0,0])
        [w, iters] = pla(X,Y,w0)
        pla_iters.append(iters)
        w = pseudoinverse(X, Y)
        [w, iters] = pla(X,Y,w)
        pla_pseudo_iters.append(iters)
    print 'avg # Iterations without Pseudoinverse', np.average(pla_iters)              #Taking average
    print 'avg # Iterations with Pseudoinverse', np.average(pla_pseudo_iters)          #Taking average

    '''
    #Lines for plotting graph   
    color_array = np.array(['r','b','g'])
    plt.scatter(X[:,0],X[:,1],s=50,c=color_array[Y+1])
    plt.plot(line_points[:,0],line_points[:,1],marker='o',linestyle= '--',color='r')
    plt.show()
    '''