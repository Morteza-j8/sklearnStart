# Load the data
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cluster as cl
import numpy as np
import itertools


# Generate a dataset and plot it
two_moon_x, two_moon_y = sklearn.datasets.make_moons(500, 0, noise=0.035)
# two_ring_x, two_ring_y = sklearn.datasets.make_circles(500, 0, noise=0.15, factor=0.2)


# plt.scatter(two_ring_x[:, 0], two_ring_x[:, 1], s=25, c=two_ring_y, cmap=plt.cm.Spectral, marker='o')
# plt.scatter(two_moon_x[:, 0], two_moon_x[:, 1], s=50, c=two_moon_y, cmap=plt.cm.Spectral, marker='x')

# data = np.arra




#[ [1,  2], [1,  4], [1,  0],
#  [10, 2], [10, 4], [10, 0]
#]


# for u in range(0 , len(two_moon_x))




kmeans = cl.KMeans(n_clusters=2, random_state=0).fit(two_moon_x)
labels = kmeans.labels_
predicate = kmeans.predict([[0, 0], [12, 3]])
cl_center = kmeans.cluster_centers_

matplotlib.style.use('ggplot') #makes plots look pretty
plt.scatter(two_moon_x[:,0],two_moon_x[:,1], s = 30 , c = labels, cmap=plt.cm.Spectral)
plt.show()




