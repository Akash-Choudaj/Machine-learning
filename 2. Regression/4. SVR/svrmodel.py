import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[: ,  -1].values

# from  sklearn.model_selection import train_test_split
# x_train, x_test, y_train,y_test  =train_test_split(x,y , train_size=0.2, random_state=1)

y = y.reshape(len(y), 1) # to transform into 2 d array

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


