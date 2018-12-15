########################
#    Written By: Avinab Saha, 15EC10071
#    Mies Coding Assignment on Multi Layered Perceptron (3,4,1)
########################

import numpy as np

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def ReLU(x):
    return x * (x > 0)

def derivatives_sigmoid(x):
	return x * (1 - x)

def derivatives_ReLU(x):
	return 1 * (x>0)

# Training Data
X=np.array([[1.81,0.80,0.44],[1.77,0.70,0.43],[1.60,0.60,0.38],[1.54,0.54,0.37],[1.66,0.65,0.40],
	[1.90,0.90,0.47],[1.75,0.64,0.39],[1.77,0.70,0.40],[1.59,0.55,0.37],
	[1.71,0.75,0.42],[1.81,0.85,0.43]])

#Training Labels
y=np.array([[0],[0],[1],[1],[0],[0],[1],[1],[1],[0],[0]])

#Test_Data
X_test=np.array([[1.63, 0.60, 0.37],[1.75, 0.72, 0.41]])

#Initialization of hyperparameters
epoch=30000
lr=2e-2
inputlayer_neurons = X.shape[1] 
hiddenlayer_neurons = 4  
output_neurons = 1 

#Initialization of weight and bias
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))


#Training Phase
for i in range(epoch):

	#Forward Propogation
	hidden_layer_input1=np.dot(X,wh)
	hidden_layer_input=hidden_layer_input1 + bh
	hiddenlayer_activations = ReLU(hidden_layer_input)

	output_layer_input1=np.dot(hiddenlayer_activations,wout)
	output_layer_input= output_layer_input1+ bout
	output = sigmoid(output_layer_input)

	#Backpropagation
	E = y-output
	slope_output_layer = derivatives_sigmoid(output)
	slope_hidden_layer = derivatives_ReLU(hiddenlayer_activations)
	d_output = E * slope_output_layer
	Error_at_hidden_layer = d_output.dot(wout.T)
	d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
	wout += hiddenlayer_activations.T.dot(d_output) *lr
	bout += np.sum(d_output, axis=0,keepdims=True) *lr
	wh += X.T.dot(d_hiddenlayer) *lr
	bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr


# Test phase

hidden_layer_input1=np.dot(X_test,wh)
hidden_layer_input=hidden_layer_input1 + bh
hiddenlayer_activations = ReLU(hidden_layer_input)

output_layer_input1=np.dot(hiddenlayer_activations,wout)
output_layer_input= output_layer_input1+ bout
output_test = sigmoid(output_layer_input)

#Thresholding Outputs with 0.50
output_pred=[]
output_pred_test=[]
for o in output:
	if(o>0.5):
		output_pred.append(1)
	if(o<=0.5):
		output_pred.append(0)


# Print Raw regressed scores for train data
print("Predictions For Train Data")
print("Raw Outputs for train data")
print output[0][0],output[1][0],output[2][0],output[3][0],output[4][0],output[5][0],output[6][0],output[7][0],output[8][0],output[9][0],output[10][0]
# Print Class Labels for test data
print("Class Labels for train data")
print output_pred[0],output_pred[1],output_pred[2],output_pred[3],output_pred[4],output_pred[5],output_pred[6],output_pred[7],output_pred[8],output_pred[9],output_pred[10]

count=0
for i in range(11):
	#print(y[i][0])
	if (output_pred[i]==y[i][0]):
		count=count+1

print('Training Accuracy:'),
print((float(count)/float(11)*100))

for o in output_test:
	if(o>0.5):
		output_pred_test.append(1)
	if(o<=0.5):
		output_pred_test.append(0)

# Print Raw regressed scores for test data
print("Predictions for Test Data")
print("Raw Outputs for test data")
print output_test[0][0],output_test[1][0]
# Print Class Labels for test data
print("Class Labels for test data")
print output_pred_test[0],output_pred_test[1]
