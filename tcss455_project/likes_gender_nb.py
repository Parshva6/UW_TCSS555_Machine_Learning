import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


# Merge relation.csv and profile.csv by userid
# Likes of each user are combined to a string separated by commas
def user_likes_match():
    # rReading profile.csv
    profile = pd.read_csv("training/profile/profile.csv")
    # Reading relation.csv, converting float to str
    likes = pd.read_csv("training/relation/relation.csv", converters={'like_id': lambda x: str(x)})

    # Combine the likes of each user to a string separated by commas
    likes["like_id"] = likes.groupby("userid")["like_id"].transform(lambda x: ','.join(x))
    likes = likes.drop_duplicates(subset="userid")

    # Match profile and likes by userid
    likes_gender_df = pd.merge(likes, profile, on="userid")
    likes_gender_df.drop(likes_gender_df.columns[[0, 3]], axis=1, inplace=True)

    # Output as .csv file
    likes_gender_df.to_csv("user_likes.csv")
    return likes_gender_df


# Train a NB model predicting gender by likes
def nb_train_likes_gender():
    # loading data
    data = user_likes_match()
    data.drop(columns=["age", "ope", "con", "ext", "agr", "neu"], axis=1, inplace=True)

    # Split data into train and test
    trainingSet, testSet = train_test_split(data, test_size=0.2)

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

if __name__ == "__main__":
    nb_train_likes_gender()
