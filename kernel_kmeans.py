# Load the data
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cluster as cl
import numpy as np


def distance(p1, p2):
    np.linalg.norm(p1-p2)



data, label = sklearn.datasets.make_circles(500, 0, noise=0.15, factor=0.2)

K = np.zeros((len(data) , len(data)))



for i in range(len(data)):
    for j in range(len(data)):
        K[i][j] = distance(data[i], data[j])




kmeans = cl.KMeans(n_clusters=2, random_state=0).fit(two_moon_x)
labels = kmeans.labels_
predicate = kmeans.predict([[0, 0], [12, 3]])
cl_center = kmeans.cluster_centers_

matplotlib.style.use('ggplot') #makes plots look pretty
plt.scatter(two_moon_x[:,0],two_moon_x[:,1], s = 30 , c = labels, cmap=plt.cm.Spectral)
plt.show()




