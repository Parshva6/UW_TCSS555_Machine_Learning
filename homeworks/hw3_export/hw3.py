# course: TCSS455
# ML in Python, homework 3
# date: 13/05/2019
# name: Martine De Cock
# description: Neural network for predicting personality of Facebook users

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Loading the data
# There are 9500 users (rows)
# There are 81 columns for the LIWC features followed by columns for
# openness, conscientiousness, extraversion, agreeableness, neuroticism
# As the target variable, we select the extraversion column (column 83)
dataset = np.loadtxt("Facebook-User-LIWC-personality-HW3.csv", delimiter=",")
X = dataset[:,0:81]
y = dataset[:,83]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

# Training and testing a linear regression model
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('MSE with linear regression:', metrics.mean_squared_error(y_test, y_pred))

# Training and testing a neural network
model = Sequential()
model.add(Dense(3, input_dim=81, kernel_initializer='normal', activation='relu'))
model.add(Dense(2, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X_train,y_train, epochs=100)
y_pred = model.predict(X_test)
print('MSE with neural network:', metrics.mean_squared_error(y_test, y_pred))