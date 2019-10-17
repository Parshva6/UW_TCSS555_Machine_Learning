import random
import sys
import csv
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as et
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter

input_directory = sys.argv[1]
output_directory = sys.argv[2]

#os.remove("df_relation_test.csv")
#os.remove("df_trained_upu.csv")
#os.remove("likes.csv")

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

#Command line args and setting up paths
profile_test = pd.read_csv(input_directory + "profile/profile.csv",index_col=0)
relation_test = pd.read_csv(input_directory + "relation/relation.csv",index_col=0)

df_relation_test = relation_test.copy()
df_relation_test = df_relation_test.groupby(by='like_id', as_index=False).agg({'userid': pd.Series.nunique})
df_relation_test = df_relation_test.rename(columns={'userid': 'count'})

likes = relation_test.copy()
likes['userid'] = likes['userid'].astype(str)
likes = likes['userid'].str.split(', ').groupby(likes['like_id']).agg(lambda x: ', '.join(set(y for z in x for y in z))).reset_index()
final = pd.merge(df_relation_test, likes, on="like_id")
    
# Take in only pages with more than 2 likes
final = final.loc[final['count'] > 10]
final = final.loc[final['count'] < 700]
final.to_csv("final.csv")



'''
merge_file = pd.merge(left=profile_test, right=final, left_on='userid', right_on='userid')       
merge_file['like_id'] = merge_file['like_id'].astype(str)
merge_file['like_id'] = merge_file['like_id'] + " "
pageIDs = merge_file.groupby('userid')['like_id'].apply(lambda x: x.sum()).reset_index()  
df_trained_upu = pd.merge(left=profile_test, right=pageIDs, left_on='userid', right_on='userid')
df_trained_upu.to_csv("df_trained_upu.csv")
#print(df_trained_upu)

# Split data into train and test
trainingSet, testSet = train_test_split(df_trained_upu, test_size=0.2)

# Train a Naive Bayes model
cv = CountVectorizer()
x_train = cv.fit_transform(trainingSet["like_id"])
y_train = trainingSet["gender"]
clf = MultinomialNB()
clf.fit(x_train, y_train)


# Test a Naive Bayes model
x_test = cv.transform(testSet["like_id"])
y_test = testSet["gender"]
y_predicted = clf.predict(x_test)

# Clean created file
#os.remove("user_likes.csv")

# Output the prediction accuracy
print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))
'''



