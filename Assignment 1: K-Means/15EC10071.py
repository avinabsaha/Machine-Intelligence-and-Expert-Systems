# Created By: Avinab Saha 15EC10071
# MIES Computer Assignment 1


import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import math


def perform_kmeans(Data_points,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters,random_state=50) 
    # Fit the model 
    kmeans.fit(Data_points)  

    # Clustered Points Colored
    plt.scatter(Data_points[:,0],Data_points[:,1], c=kmeans.labels_, cmap='rainbow') 
    #plt.show()
    # Save Plot
    s = "Clustered Points for k="+str(n_clusters)
    plt.grid(True)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(s)
    print("Cluster Centroids are")
    print(kmeans.cluster_centers_) 
    length = len(kmeans.labels_)
    # Uncomment to print class labels for all points
    #print(len(kmeans.labels_))

    # Computer error
    error = 0
    for i in range(length):
        x_error = math.pow(Data_points[i,0]-(kmeans.cluster_centers_[kmeans.labels_[i]])[0],2)
        y_error = math.pow(Data_points[i,1]-(kmeans.cluster_centers_[kmeans.labels_[i]])[1],2)
        error = error + math.sqrt(x_error+y_error)

    s2 = "Error for k="+str(n_clusters)+" is "
    print(s2+str(error/length))
    return error/length
                    


def main():
    dfs = pd.read_excel('dataset.xlsx',header=None)
    Data_points = dfs.values

    # Original Points
    plt.scatter(Data_points[:,0],Data_points[:,1], label='True Position') 
    plt.grid(True)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    #plt.show()
    plt.savefig('Plot Original Points')

    errorK = np.zeros(5)
    for loop in range(5):
        st = "Performing K-Means Clustering with k="+str(loop+1)
        print(st)
        errorK[loop] = perform_kmeans(Data_points,n_clusters=loop+1)

    plt.figure(0)
    arr = np.array([1,2,3,4,5])
    plt.plot(arr, errorK, linewidth=2.0)
    plt.grid(True)
    plt.xlabel('Increasing Value of K')
    plt.ylabel('Error')
    plt.savefig('Error change with Increasing K')

if __name__ == "__main__":
    main()