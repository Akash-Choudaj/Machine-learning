# importing libraries and dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("Position_Salaries.csv")

# Spliting the data
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Decisssion Tree for whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(x, y)

# Model predicting for value 6.5
regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results

# Create many values between minimum and maximum x
X_grid = np.arange(min(x[:, 0]), max(x[:, 0]), 0.01)

# Convert X_grid into 2D array because the model expects 2D input
X_grid = X_grid.reshape((len(X_grid), 1))

# Original data points
plt.scatter(x, y, color='red')

# Decision Tree predictions
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()