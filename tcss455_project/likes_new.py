import sys
import csv
import os
import warnings
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from collections import Counter

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

input_directory = sys.argv[1]
output_directory = sys.argv[2]

#Load dataframes
profile = pd.read_csv(input_directory + 'profile/profile.csv',index_col=0)
relation = pd.read_csv(input_directory + 'relation/relation.csv',index_col=0)
relation.like_id = relation.like_id.astype(str)
train_profile = pd.read_csv('training/profile/profile.csv',index_col=0)
train_relation = pd.read_csv('training/relation/relation.csv',index_col=0)
train_relation.like_id = train_relation.like_id.astype(str)

###############################################################################################
# Texts: LR on gender and age

train_profile['text'] = ''
index = 0

for row in train_profile['userid']:
    text = row + '.txt'
    text = open('training/text/' + text, 'rt',encoding='ISO-8859-1')
    line = text.read()
    train_profile.set_value(index, 'text', line)
    index+=1
    text.close()

profile['text'] = ''
index = 0

for row in profile['userid']:
    text = row + '.txt'
    text = open(input_directory + 'text/' + text, 'rt',encoding='ISO-8859-1')
    line = text.read()
    profile.set_value(index, 'text', line)
    index+=1
    text.close()

# Add age_group column to profile dataframe
train_profile['age_group'] = 0
train_profile['age_group'].loc[train_profile['age']<25]=0
train_profile['age_group'].loc[(train_profile['age']>=25) & (train_profile['age']<35)]=1
train_profile['age_group'].loc[(train_profile['age']>=35) & (train_profile['age']<50)]=2
train_profile['age_group'].loc[train_profile['age']>=50]=3

gender_train1 = train_profile.loc[:,['text', 'gender']]
gender_test1 = profile.loc[:,['text', 'gender']]
age_train1 = train_profile.loc[:,['text', 'age_group']]
age_test1 = profile.loc[:,['text', 'age_group']]

# Training Naive Bayes model
cv1 = CountVectorizer()
gender_train_x1 = cv1.fit_transform(gender_train1['text'])
gender_train_y1 = gender_train1['gender']
age_train1_x = cv1.fit_transform(age_train1['text'])
age_train1_y = age_train1['age_group']

clf1_gender = LogisticRegression()
clf1_gender.fit(gender_train_x1, gender_train_y1)
clf1_age = LogisticRegression()
clf1_age.fit(age_train1_x, age_train1_y)

# Predict test data
gender_test_x1 = cv1.transform(gender_test1['text'])
gender_LR_text = clf1_gender.predict(gender_test_x1)
age_test_x1 = cv1.transform(age_test1['text'])
age_LR_text = clf1_age.predict(age_test_x1)

###############################################################################################

total = train_relation['like_id'].value_counts()
a = train_relation[train_relation['like_id'].isin(total[total < 900].index & total[total > 1].index)]

# training data frame to apply NB and LR on likes
grouped = a.groupby('userid',as_index=False).agg({'like_id': lambda x: '%s' % ' '.join(x)})
merged_train = (pd.merge(train_profile, grouped, left_on='userid', right_on='userid'))
grouped2 = relation.groupby('userid',as_index=False).agg({'like_id': lambda x: '%s' % ' '.join(x)})
merged_test = (pd.merge(profile, grouped2, left_on='userid', right_on='userid'))

# Likes: NB on age and gender

gender_train2_y = merged_train['gender']
age_train2_y = merged_train['age_group']

cv2 = CountVectorizer()
age_train_x2 = cv2.fit_transform(merged_train['like_id'])
age_test_x2 = cv2.transform(merged_test['like_id'])
gender_train_x2 = cv2.fit_transform(merged_train['like_id'])
gender_test_x2 = cv2.transform(merged_test['like_id'])

clf2_gender = MultinomialNB()
clf2_gender.fit(gender_train_x2, gender_train2_y)
clf2_age = MultinomialNB()
clf2_age.fit(age_train_x2, age_train2_y)

gender_NB_likes = clf2_gender.predict(gender_test_x2)
age_NB_likes = clf2_age.predict(age_test_x2)

###############################################################################################

# Likes: LR on age and gender

gender_train3 = merged_train['gender']
age_train3 = merged_train['age_group']

cv3 = CountVectorizer()
likes_train = cv3.fit_transform(merged_train['like_id'])
likes_test = cv3.transform(merged_test['like_id'])

clf3_gender = LogisticRegression()
clf3_gender.fit(likes_train, gender_train3)
clf3_age = LogisticRegression()
clf3_age.fit(likes_train, age_train3)

gender_LR_likes = clf3_gender.predict(likes_test)
age_LR_likes = clf3_age.predict(likes_test)

###############################################################################################

gender_values = []
age_values = []
train_profile = input_directory + 'profile/profile.csv'

for a, b, c in zip(age_LR_likes, age_NB_likes, age_LR_text):
    temp = [a,b,c]
    count = Counter(temp)
    i = count.most_common()[0][0]
    if i==0:
        age_values.append('xx-24')
    elif i==1:
        age_values.append('25-34')
    elif i==2:
        age_values.append('35-49')
    else:
        age_values.append('50-xx')

for a, b in zip(gender_LR_text, gender_LR_likes):
    temp = [a, b]
    if temp==[0.0, 0.0]:
        gender_values.append('male')
    else:
        gender_values.append('female')

#Parsing through the test data
with open(train_profile) as csvfile:
        testreader = csv.reader(csvfile,delimiter=',')
        header = next(testreader)
        index = 0

        for row in testreader:
            root = et.Element('user', id=row[1], age_group=age_values[index],
                              gender=gender_values[index], extrovert='3.49', 
                              neurotic='2.73', agreeable='3.58', 
                              conscientious='3.45', open='3.91')

            tree = et.ElementTree(root)
            folder_name = '/Users/macbook/Downloads/output/' + row[1] + '.xml'
            tree.write(folder_name)
            index += 1

csvfile.close();




