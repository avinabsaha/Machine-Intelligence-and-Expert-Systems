"""
Code Written By: Avinab Saha, 15EC10071
Mies Assignment 6: Naive Bayes Classifier
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
dataset = datasets.load_iris()

Label = dataset['target']
Data = dataset['data']

model = GaussianNB()
model.fit(Data,Label)

y_pred = model.predict(Data)
score = (y_pred == Label).sum()

print("Total {} instances".format(Label.shape[0]))
print("Correctly Classified {} instances".format(score))
print("Incorrectly Classified {} instances".format(Label.shape[0]-score))

print("Overall Accuracy:{}%".format(float(100*score)/Label.shape[0]))

confusion_matrix  = confusion_matrix(Label, y_pred)
print("Confusion Matrix:")
print(confusion_matrix)

class_0 = (Label == 0).sum()
class_1 = (Label == 1).sum()
class_2 = (Label == 2).sum()


# For Class 0
tp_0 = confusion_matrix[0,0]
fp_0 = confusion_matrix[0,1]+confusion_matrix[0,2]
fn_0 = confusion_matrix[1,0]+confusion_matrix[2,0]
precision_0 = float(tp_0)/float(tp_0+fp_0)
recall_0 = float(tp_0)/float(tp_0+fn_0)
F1_0 = (2*precision_0*recall_0)/(precision_0+recall_0)
print("Details for Class Labelled 0")
print("Accuracy:{}%".format(float(100*confusion_matrix[0,0])/class_0))
print("Precision:{}".format(precision_0))
print("Recall:{}".format(recall_0))
print("F1 Score: {}".format(F1_0))

# For Class 1
tp_1 = confusion_matrix[1,1]
fp_1 = confusion_matrix[1,0]+confusion_matrix[1,2]
fn_1 = confusion_matrix[0,1]+confusion_matrix[2,1]
precision_1 = float(tp_1)/float(tp_1+fp_1)
recall_1 = float(tp_1)/float(tp_1+fn_1)
F1_1 = (2*precision_1*recall_1)/(precision_1+recall_1)
print("Details for Class Labelled 1")
print("Accuracy:{}%".format(float(100*confusion_matrix[1,1])/class_1))
print("Precision:{}".format(precision_1))
print("Recall:{}".format(recall_1))
print("F1 Score: {}".format(F1_1))

# For Class 2
tp_2 = confusion_matrix[2,2]
fp_2 = confusion_matrix[2,0]+confusion_matrix[2,1]
fn_2 = confusion_matrix[0,2]+confusion_matrix[1,2]
precision_2 = float(tp_2)/float(tp_2+fp_2)
recall_2 = float(tp_2)/float(tp_2+fn_2)
F1_2 = (2*precision_2*recall_2)/(precision_2+recall_2)
print("Details for Class Labelled 2")
print("Accuracy:{}%".format(float(100*confusion_matrix[2,2])/class_2))
print("Precision:{}".format(precision_2))
print("Recall:{}".format(recall_2))
print("F1 Score: {}".format(F1_2))
