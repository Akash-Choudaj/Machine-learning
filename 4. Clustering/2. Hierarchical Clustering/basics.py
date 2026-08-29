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

# Create the Hierarchical Clustering model
hc = AgglomerativeClustering(
    n_clusters=3,          # Number of clusters we want
    affinity='euclidean',  # Distance method used to measure similarity
    linkage='ward'         # Method used to decide which clusters to merge
)

# Train the model and assign each data point to a cluster
y_hc = hc.fit_predict(x)