# course: TCSS555
# ML in Python
# date: 04/09/2018
# name: Martine De Cock
# description: Decision tree for predicting heart disease
# The dataset contains 303 patients who presented with chest pain.
# For every patient, there are 
#     13 predictors including age, sex, chol (cholesterol measurement) and other heart and lung function measurements.
#     a binary outcome value: 
#         'Yes' indicates the presence of heart disease based on angiographic test 
#         'No' means no heart disease

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import tree



from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve

# Loading the data

data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Heart.csv", index_col=0)

print(data.head())
#print(data.describe())

# Preprocessing: removing rows with NaN, and mapping categorical input features to numbers
print("Number of rows in data:", data.shape[0])
data = data.dropna()

print("Number of rows remaining after removing the patients with a NaN field:", data.shape[0])



map_to_int = {'typical':1, 'nontypical':2, 'nonanginal':3, 'asymptomatic':4}
data['ChestPain'] = data['ChestPain'].replace(map_to_int)

map_to_int = {'fixed':1, 'normal':2, 'reversable':3}
data['Thal'] = data['Thal'].replace(map_to_int)
print(data.head())

# Training and testing a decision tree
y = data.AHD


X = data.drop('AHD', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = tree.DecisionTreeClassifier(max_depth=3)

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)


print("Accuracy: %.2f" % accuracy_score(y_test,y_pred))






# Computing AUC



y_prob = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label='Yes', drop_intermediate='False')

roc_auc = auc(fpr, tpr)

print("AUC: %.2f" % roc_auc)