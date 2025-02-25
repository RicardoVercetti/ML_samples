import matplotlib
matplotlib.use('TkAgg')  # General backend for plots
 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
 
# Load dataset
df = pd.read_csv("/home/jehoniah/Documents/Personel/Repos/ML_samples/GFG/Mod2-SupervisedLearning/res/Housing.csv")

# Extract features and target variable
Y = df['price']
X = df['lotsize']


# Reshape for compatibility with scikit-learn
X = X.to_numpy().reshape(len(X), 1)
Y = Y.to_numpy().reshape(len(Y), 1)

# Split data into training and testing sets
X_train = X[:-250]
X_test = X[-250:]
Y_train = Y[:-250]
Y_test = Y[-250:]

# Plot the test data
plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

# Train linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

# Plot predictions
plt.plot(X_test, regr.predict(X_test), color='red', linewidth=3)
plt.show()


# print("Predicted price for a lot size of 5000: " + str(round(regr.predict([[5000]])[0][0])))