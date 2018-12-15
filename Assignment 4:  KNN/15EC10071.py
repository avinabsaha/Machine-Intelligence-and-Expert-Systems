########################
#    Written By: Avinab Saha, 15EC10071
#    Mies Coding Assignment on KNN
########################


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# Reading number of Training Examples

def most_common(lst):
    return max(set(lst), key=lst.count)

with open('iris.csv','r') as f:
	data = []
	for l in f.readlines():
		data.append([str(x) for x in l.strip('\r\n').split(',')])
	f.close()

del(data[0])

for loop in range(len(data)):
    for loop2 in range(4):
        data[loop][loop2] = float(data[loop][loop2])


tags = []
for loop in range(len(data)):
    tags.append(data[loop][4])
    del(data[loop][4])

data = np.asarray(data)
#print(tags)
elements = list(set(tags))
#print(elements)


for count in range(len(data)):
    if (tags[count]==elements[0]):
        tags[count]= int(0)
    if (tags[count]==elements[1]):
        tags[count]= int(1)
    if (tags[count]==elements[2]):
        tags[count]= int(2)
    count=count+1


tags = np.asarray(tags)

#print(data.shape)
#print(tags.shape)

test = np.array([7.2, 3.6, 5.1, 2.5])


dist=np.zeros(len(data))
for loop in range(len(data)):
    dist[loop]= np.sum(np.square(data[loop]-test))

#print(dist)


for k in range(1,6):
	print("For k equals:"),
	print(k)
	idx = dist.argsort()[:k]
	#print(idx)

	outputs = []
	for i in range(k):
    		outputs.append(elements[tags[idx[i]]])

	#print(outputs)
	print("Prediction using Own Implementation"),
	print(most_common(outputs))



	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(data,tags) 
	test = test.reshape(1,-1)
	print("Prediction using Sklearn Implementation"),
	print(elements[neigh.predict(test)[0]])
