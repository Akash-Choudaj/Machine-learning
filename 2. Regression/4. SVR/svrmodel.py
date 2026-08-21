import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Convert y into a 2D array because StandardScaler expects 2D input
y = y.reshape(len(y), 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# SVR Training
from sklearn.svm import SVR

# Using RBF (Radial Basis Function) kernel
# RBF kernel is useful for non-linear relationships
regressor = SVR(kernel='rbf')

# Training the model
# fit() = learn/train the relationship between x and y
# ravel() converts y from 2D to 1D because SVR expects a 1D target
regressor.fit(x, y.ravel())

# Predicting the new data
# 6.5 is the new value we want to predict
# sc_x.transform() scales the new input using the same scaling learned from x
# predict() makes the prediction
# sc_y.inverse_transform() converts the prediction back to the original salary scale
pred = sc_y.inverse_transform(
    regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1)
)

# print(pred)

# Visualizing the model
plt.scatter(
    sc_x.inverse_transform(x),
    sc_y.inverse_transform(y),
    color='red'
)

plt.plot(
    sc_x.inverse_transform(x),
    sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)),
    color='blue'
)

plt.title('SVR Model')
plt.show()