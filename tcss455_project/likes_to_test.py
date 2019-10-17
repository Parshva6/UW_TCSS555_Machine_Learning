import os
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import sklearn.linear_model as sk
import xml.etree.ElementTree as et


# Merge relation.csv and profile.csv by userid
def user_likes_match(file_source):

    # Identify which ones to merge, training or testing files
    if file_source == 'training':
        likes = training_likes
        profile = training_profile

    elif file_source == 'testing':
        likes = testing_likes
        profile = testing_profile

    # Combine the likes of each user to a string separated by commas
    likes["like_id"] = likes.groupby("userid")["like_id"].transform(lambda x: ','.join(x))
    likes = likes.drop_duplicates(subset="userid")

    # Match profile and likes by userid
    likes_gender_df = pd.merge(likes, profile, on="userid")
    likes_gender_df.drop(likes_gender_df.columns[[0, 3]], axis=1, inplace=True)

    # Output as .csv file
    likes_gender_df.to_csv(file_source + "_user_likes.csv")


# Train a LR model predicting gender by likes
def lr_train_likes_gender():
    # Loading data
    df_training = pd.read_csv("training_user_likes.csv")
    train_data = df_training.loc[:,['gender', 'like_id']]
    df_testing = pd.read_csv("testing_user_likes.csv")
    testing_data = df_testing.loc[:,['gender', 'like_id']]

    # Split data into train and test
    #trainingSet, testSet = train_test_split(data, test_size=0.2)
    n = 1500
    total = np.arange(len(train_data)) 
    #random.shuffle(all_Ids)
    test_Ids = total[0:n]
    train_Ids = total[n:]
    data_test = train_data.loc[test_Ids, :]
    data_train = train_data.loc[train_Ids, :]

    # Train a linear regression model
    cv = CountVectorizer()
    x_train = cv.fit_transform(data_train["like_id"])
    y_train = data_train["gender"]
    clf = sk.LogisticRegression()
    clf.fit(x_train, y_train)

    # Test a Naive Bayes model
    x_test = cv.transform(data_test["like_id"])
    y_test = data_test["gender"]
    y_predicted = clf.predict(x_test)

    X_test_data = cv.transform(testing_data['like_id'])
    #y_test_in_test_data = test_targets['gender']
    y_predicted_data = clf.predict(X_test_data)

    df_testing['gender'] = y_predicted_data
    test_gender_userid = df_testing.loc[:,['userid', 'gender']]
    test_gender_userid.to_csv("gender_prediction.csv")

    # Output the prediction accuracy
    print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))

    # Clean temp file
    #os.remove("training_user_likes.csv")
    #os.remove("testing_user_likes.csv")


#Output corresponding xml files to each userid in a folder called ouput
def output():
    # Loading data
    #file = pd.read_csv(input_path + "profile/profile.csv")
    file = pd.read_csv("gender_prediction.csv")
    # Row count is to count the total of test target
    row_count = 0
    testTargetIDArray = []
    testTargetGenderArray = []
    price = 0

    # Get all test target id into an array by using for loop
    for index, row in file.iterrows():
        testTargetID = row['userid']
        testTargetGender = row['gender']
        if(testTargetGender == 1):
            testTargetGender = 'female'
        else:
            testTargetGender = 'male'
            price +=1

        testTargetIDArray.append(testTargetID)
        testTargetGenderArray.append(testTargetGender)
        row_count += 1

    print(price)

    # Output xml files for each of the users
    for row in range(row_count):
        root = et.Element('user', id=testTargetIDArray[row], age_group='xx-24',
                          gender=testTargetGenderArray[row], extrovert='3.49', 
                          neurotic='2.73', agreeable='3.58', 
                          conscientious='3.45', open='3.91')

        tree = et.ElementTree(root)
        folder_name = "/Users/macbook/Downloads/" + output_path + testTargetIDArray[row] + ".xml"
        tree.write(folder_name)

    # Clean temp file
    #os.remove("gender_prediction.csv")


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    base_path = os.path.dirname(os.path.realpath(__file__))

    # Reading profile.csv
    training_profile = pd.read_csv("training/profile/profile.csv")
    # Reading relation.csv, converting float to str
    training_likes = pd.read_csv("training/relation/relation.csv", converters={'like_id': lambda x: str(x)})
    # Reading profile.csv
    testing_profile = pd.read_csv(input_path + "profile/profile.csv")
    # Reading relation.csv, converting float to str
    testing_likes = pd.read_csv(input_path + "relation/relation.csv", converters={'like_id': lambda x: str(x)})

    #user_likes_match(training_likes, training_profile)
    #testing_data_merge(testing_likes, testing_profile)
    user_likes_match('training')
    user_likes_match('testing')
    lr_train_likes_gender()
    output()





