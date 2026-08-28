import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:, [3, 4]].values # took last to row because to visualize the cluster in 2D plane properly other wise we cantake every row in dataset but it will be complex to visualize

# Elbow method to find optimal number of clusters
from sklearn.cluster import KMeans  # Import the KMeans clustering algorithm from scikit-learn
wcss = []  # Create an empty list to store WCSS values
for i in range(1, 11):  # Run the loop for 1 to 10 possible clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # Create a KMeans model with i clusters
    kmeans.fit(x)  # Train the KMeans model using the dataset x
    wcss.append(kmeans.inertia_)  # Store the WCSS value of the current number of clusters
plt.plot(range(1, 11), wcss)  # Plot number of clusters against their WCSS values
plt.title('The Elbow Method')  
plt.show()  