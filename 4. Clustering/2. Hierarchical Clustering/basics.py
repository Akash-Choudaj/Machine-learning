import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(('Mall_Customers.csv'))

x = dataset.iloc[:, [3,4]].values

# Dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendgrom = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Training Hierarchical model on dataset
# Import Agglomerative Clustering algorithm
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import AgglomerativeClustering

# Create the Hierarchical Clustering model
hc = AgglomerativeClustering(
    n_clusters=3,          # Number of clusters
    metric='euclidean',    # Distance used to measure similarity
    linkage='ward'         # Method used to merge clusters
)

# Fit the model and assign each data point to a cluster
y_hc = hc.fit_predict(x)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()