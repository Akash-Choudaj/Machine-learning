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