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

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()