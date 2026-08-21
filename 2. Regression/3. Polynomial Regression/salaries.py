import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Linear regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x,y)

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
linreg2 = LinearRegression()
linreg2.fit(x_poly, y)

# visulation of linear regression
plt.scatter(x,y)
plt.plot(x, linreg.predict(x))
plt.show()


# visulation of polynomial regression
plt.scatter(x,y)
plt.plot(x, linreg2.predict(poly_reg.fit_transform(x)))
plt.show()
