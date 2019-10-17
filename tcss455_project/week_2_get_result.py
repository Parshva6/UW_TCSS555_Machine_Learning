#jatt
import csv
import math
import os
import xml.etree.ElementTree as et
import sys
import codecs
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

input_path = sys.argv[1]
output_path = sys.argv[2]

# array for user id, age, gender
# and five persionality - 
# openness(ope), conscientiousness(con)
# extroversion(ext), agreeableness(agr)
# emotional stability(neu)
userIDArray = []
ageArray = []
genderArray = []
opeArray = []
conArray = []
extArray = []
agrArray = []
neuArray = []

# array for user id in txt file and 
# content in txt file corresponding 
# with user id.
txtIDArray = []
txtContentArray = []


testUserIDArray = []
testTxTIDArray = []
testTxtContentArray = []




# width and height for tranningList
w, h = 9, 9500

# training list is a 2D array
trainingList = [[0 for x in range(w)] for y in range(h)]

# width and height txtList
w_for_text_files, h_for_text_files = 2, 9500

# txt list is a 2D array
txtList = [[0 for x in range(w_for_text_files)] for y in range(h_for_text_files)]


# the header is use for output csv file
header = ['userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu', 'text']

header_test = ['userid', 'gender','text']


# # base path is the path for the directory
# base_path_for_original_profile_csv = open('/Users/navdeepkhatra/Desktop/tcss455/hw1/profile.csv')
# base_path_for_training_data_text_folder = '/Users/navdeepkhatra/Desktop/tcss455/training/text/'

# base path is the path for the directory of profile.csv
# base_path_for_original_profile_csv = open('C:/Users/BunBun/Documents/TCSS 455 Machine Learning/second time for project/profile.csv')
base_path_for_original_profile_csv = open('training/profile/profile.csv')

# base_path_for_test_profile_csv = open('C:/Users/BunBun/Documents/TCSS 455 Machine Learning/second time for project/profile_test.csv')
base_path_for_test_profile_csv= open(input_path + 'profile/profile.csv')

# base path is the path for the text files of the training data
# base_path_for_training_data_text_folder = 'C:/Users/BunBun/Documents/TCSS 455 Machine Learning/second time for project/training_text/'
base_path_for_training_data_text_folder = 'training/text/'

# base_path_for_test_data_text_folder = 'C:/Users/BunBun/Documents/TCSS 455 Machine Learning/second time for project/test_text/'
base_path_for_test_data_text_folder = input_path + 'text/'


def takeFirst(elem):
    return elem[0]

# open training data csv file 
with base_path_for_original_profile_csv as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ',')

     # skip the first row
    next(csvfile)

    # row count is to count the total of training target
    row_count = 0

    # Get all training  target info into arrays by using for loop
    # and increment the row_count variable
    for row in readCSV:
        userid = row[1]
        age = row[2]
        gender = row[3]
        ope = row[4]
        con = row[5]
        ext = row[6]
        agr = row[7]
        neu = row[8]

        userIDArray.append(userid)
        ageArray.append(age)
        genderArray.append(gender)
        opeArray.append(ope)
        conArray.append(con)
        extArray.append(ext)
        agrArray.append(agr)
        neuArray.append(neu)
        row_count += 1


    # total amount of text files
    row_count_for_txt_file = 0

    # Plug in info to the training target list by using for loop
    for row in range(row_count):
        trainingList[row][0] = userIDArray[row]
        trainingList[row][1] = "%.0f" % float(ageArray[row])
        trainingList[row][2] = "%.0f" % float(genderArray[row])
        trainingList[row][3] = opeArray[row]
        trainingList[row][4] = conArray[row]
        trainingList[row][5] = extArray[row]
        trainingList[row][6] = agrArray[row]
        trainingList[row][7] = neuArray[row]
        row_count_for_txt_file += 1

    # Use for loop to go over every files in text folder
    for filename in os.listdir(base_path_for_training_data_text_folder):   

        # content is a variable to content in each text file
        content = ""

        # There is an user id in every text file
        # filename[:-4] is get rid of last four characters for the text file name 
        txtIDArray.append(filename[:-4])

        # Use codecs because there is UnicodeDecodeError in some text files 
        # errors='ignore' is the key point to slove this problem here.
        contents = codecs.open(base_path_for_training_data_text_folder + filename, encoding='utf-8', errors='ignore')

        # Go over content by line
        for line in contents:

            # Here is where content variable used for
            content += line

        # Get content in each text file to an array
        txtContentArray.append(content)

    # Now put txtIDArray and txtContentArray into txtList
    for row in range(row_count_for_txt_file):
        txtList[row][0] = txtIDArray[row]
        txtList[row][1] = txtContentArray[row]

    # Compare user ID in trainingList and user ID in txtList
    # if is matched, get the corresponding text content from txtList
    # to trainingList 
    # for row_in_training_list in range(row_count):
    #     for row_in_txt_list in range(row_count_for_txt_file):
    #         if(trainingList[row_in_training_list][0] == txtList[row_in_txt_list][0]):
    #             trainingList[row_in_training_list][8] = txtList[row_in_txt_list][1]


    trainingList.sort(key=takeFirst)
    txtList.sort(key=takeFirst)
    for row in range(row_count):
       trainingList[row][8] = txtList[row][1]
    
# write out a new profile.csv file with text column
with codecs.open('profile_modified.csv', 'w', encoding='utf-8', errors='ignore') as csvoutputfile:
    writeCSV = csv.writer(csvoutputfile, delimiter = ',')
    writeCSV.writerow(header)
    writeCSV.writerows(trainingList)
    
# close files.
csvfile.close()
contents.close()
csvoutputfile.close()

###############################################################################################

# open training data csv file 
with base_path_for_test_profile_csv as testcsvfile:
    testReadCSV = csv.reader(testcsvfile, delimiter = ',')

     # skip the first row
    next(testcsvfile)

    # row count is to count the total of test datas
    row_count_for_test_data = 0
    for row in testReadCSV:
        testUserid = row[1]
        testUserIDArray.append(testUserid)
        row_count_for_test_data += 1
    
    # create list fot test data here
    w, h = 3, row_count_for_test_data
    testDataList = [[0 for x in range(w)] for y in range(h)]

    row_count_for_test_txt_file = 0
    for row in range(row_count_for_test_data):
        testDataList[row][0] = testUserIDArray[row]
        row_count_for_test_txt_file += 1

    # for test text data
    test_text_files_count = 0
    for filename in os.listdir(base_path_for_test_data_text_folder):
        content = ""
        testTxTIDArray.append(filename[:-4])
        testTxtContents = codecs.open(base_path_for_test_data_text_folder + filename, encoding='utf-8', errors='ignore')
        for line in testTxtContents:
            content += line
        testTxtContentArray.append(content)

    w_for_test_text_files, h_for_test_text_files = 2, row_count_for_test_data
    testTxtList = [[0 for x in range(w_for_test_text_files)] for y in range(h_for_test_text_files)]

    print('testTxTIDArray: ', len(testTxTIDArray))
    print('testTxtContentArray: ', len(testTxtContentArray))
    print('testTxtList: ', len(testTxtList))
    for row in range(row_count_for_test_data):
        testTxtList[row][0] = testTxTIDArray[row]
        testTxtList[row][1] = testTxtContentArray[row]

    
    # for row_in_test_list in range(row_count_for_test_data):
    #     for row_in_test_txt_list in range(row_count_for_test_txt_file):
    #         if(testDataList[row_in_test_list][0] == testTxtList[row_in_test_txt_list][0]):
    #             testDataList[row_in_test_list][2] = testTxtList[row_in_test_txt_list][1]

    testDataList.sort(key=takeFirst)
    testTxtList.sort(key=takeFirst)
    for row in range(row_count_for_test_data):
       testDataList[row][2] = txtList[row][1]

# write out a new profile.csv file with text column
with codecs.open('profile_test_modified.csv', 'w', encoding='utf-8', errors='ignore') as testcsvoutputfile:
    writeCSV = csv.writer(testcsvoutputfile, delimiter = ',')
    writeCSV.writerow(header_test)
    writeCSV.writerows(testDataList)
    
# close files.
testcsvfile.close()
testTxtContents.close()
testcsvoutputfile.close()
###############################################################################################

# Reading the data into a dataframe and selecting the columns we need
df = pd.read_csv("profile_modified.csv")
data_Gender = df.loc[:,['gender', 'text']]

# add for profile_test_modified.csv
df_test = pd.read_csv("profile_test_modified.csv")
test_gender_text = df_test.loc[:,['gender', 'text']]

# n_test = 334
# all_test_target_Ids = np.arange(len(data_gender_data))
# test_targets_Ids = all_test_target_Ids[0:n]

# Splitting the data into 8000 training instances and 1500 test instances
n = 1500
all_Ids = np.arange(len(data_Gender)) 
#random.shuffle(all_Ids)
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = data_Gender.loc[test_Ids, :]
data_train = data_Gender.loc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['text'])
y_train = data_train['gender']
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
X_test = count_vect.transform(data_test['text'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)

X_test_in_test_data = count_vect.transform(test_gender_text['text'])
# y_test_in_test_data = test_targets_Ids['gender']
y_predicted_in_test_data = clf.predict(X_test_in_test_data)

df_test['gender'] = y_predicted_in_test_data
test_gender_userid = df_test.loc[:,['userid', 'gender']]
test_gender_userid.to_csv("profile_test_with_gender_prediction.csv")


# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
classes = [1,0]
cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
print("Confusion matrix:")
print(cnf_matrix)
# os.remove("profile_modified.csv")
# os.remove("profile_test_modified.csv")
###############################################################################################

testTargetIDArray = []
testTargetGenderArray = []

# base path is the path for the directory
base_path = os.path.dirname(os.path.realpath(__file__))

# located the test data csv file 
csv_test_file = os.path.join(base_path, 'profile_test_with_gender_prediction.csv')

# open test data csv file 
with open(csv_test_file) as csvtestfile:
    readTestCSV = csv.reader(csvtestfile, delimiter = ',')

     # skip the first row
    next(csvtestfile)

    # row count is to count the total of test target
    row_count = 0

    # Get all test target id into an array by using for loop
    # and increment the row_count variable
    # The array is for user id in test target list
    for row in readTestCSV:
        testTargetID = row[1]
        testTargetGender = row[2]
        if(testTargetGender == '1'):
            testTargetGender = 'female'
        else:
            testTargetGender = 'male'

        testTargetIDArray.append(testTargetID)
        testTargetGenderArray.append(testTargetGender)
        row_count += 1

    # output xml file
    for row in range(row_count):
        root = et.Element('user', id=testTargetIDArray[row], age_group='xx-24',
                          gender=testTargetGenderArray[row], extrovert='3.49', 
                          neurotic='2.73', agreeable='3.58', 
                          conscientious='3.45', open='3.91')

        tree = et.ElementTree(root)

        folder_name = output_path + testTargetIDArray[row] + '.xml'
        tree.write(folder_name)

        # move the file into a folder called result
        # os.rename(folder_name, base_path + "/result/" + folder_name)

# os.remove("profile_test_with_gender_prediction.csv")
print('Done')
csvtestfile.close()


