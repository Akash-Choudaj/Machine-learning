# importing libiries and dataset to train the model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Position_Salaries.csv')

# spliting the dataset
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Implementing Random Forest Tree
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state= 0) #n_estimators controls the number of tree to implement
regressor.fit(x, y)

# Preduction for new value 6.5
print(regressor.predict([[6.5]]))


# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(x[:, 0]), max(x[:, 0]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()