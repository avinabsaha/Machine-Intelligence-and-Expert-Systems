import math 
import csv
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
  
def classifyAPoint(rows,p,k=3): 
    ''' 
     This function finds classification of p using 
     k nearest neighbour algorithm. It assumes only two 
     groups and returns 0 if p belongs to group 0, else 
      1 (belongs to group 1). 
  
      Parameters -  
          points : Dictionary of training points having two keys - 0 and 1 
                   Each key have a list of training data points belong to that  
  
          p : A touple ,test data point of form (x,y) 
  
          k : number of nearest neighbour to consider, default is 3  
    '''
  
    distance=[] 
    group = []
    dist = np.zeros(150)
    i = 0
    for row in rows: 
        euclidean_distance = 0
        for feature in range(0,4):
            #print row[feature]
  
            euclidean_distance = euclidean_distance + (float(row[feature])-p[feature])**2
            #print euclidean_distance
  
        distance.append((math.sqrt(euclidean_distance),row)) 
        #dist.append((math.sqrt(euclidean_distance))
        group.append(row[4])
        dist[i] = distance[i][0]
        i=i+1
  
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    freq = np.zeros(3)
    for k in range(0, 3):
      least = min(dist[k:150])
      arg = np.argmin(dist[k:150])
      classified = group[arg]
      temp = dist[k]
      dist[k] = least
      dist[arg] = temp
      
      for i in range (0, 3):
        if classified == classes[i] :
          freq[i] = freq[i] + 1

    return classes[np.argmax(freq)]

    #print min(dist[0:100])
    #distance = sorted(distance)[:k] 
    

    #freq1 = 0 #frequency of group 0 
    #freq2 = 0 #frequency og group 1 
  
    #for d in distance: 
    #    if d[1] == 0: 
    #        freq[0] += 1
    #    elif d[1] == 1: 
    #        freq[1] += 1
  
    #return 0 if freq[0]>freq[1] else 1
  
# driver function 
def main(): 
  
    # Dictionary of training points having two keys - 0 and 1 
    # key 0 have points belong to class 0 
    # key 1 have points belong to class 1 

    
    filename = "iris.csv"
  
    # initializing the titles and rows list 
    fields = [] 
    rows = [] 
    data = [] 
    # reading csv file 
    with open(filename, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
          
        # extracting field names through first row 
        fields = csvreader.next() 
      
        # extracting each data row one by one 
        for row in csvreader: 
            rows.append(row) 

    #points = {0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)], 
    #          1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]} 
  
    # testing point p(x,y) 
    p = (7.2,3.6,5.1,2.5) 
  
    # Number of neighbours  
    for k in range(1, 6):
      print(classifyAPoint(rows,p,k))

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    neigh = KNeighborsClassifier(n_neighbors=3)
    data = np.zeros((150, 4))
    label = np.zeros(150)
    for i in range(150):
      for j in range(4):
        data[i][j] = rows[i][j]
      for c in range(3):
        if rows[i][4] == classes[c] :
          label[i] = int(c)
      

    neigh.fit(data, label) 
    #p = p.reshape(1, -1)
    #print(neigh.predict(p))
    print label

  
if __name__ == '__main__': 
    main() 
