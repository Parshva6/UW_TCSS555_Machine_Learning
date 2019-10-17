import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Loading the data
df = pd.read_csv('Facebook-User-LIWC-personality.csv', index_col=0)

# Preparing the train and test data
big5 = ['ope','ext','con','agr','neu']
X = data[big5]
LIWC_features = [x for x in df.columns.tolist()[:] if not x in big5]
X = data[feature_cols]
y = LIWC_features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
