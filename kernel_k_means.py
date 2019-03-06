# Load the data
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cluster as cl
import numpy as np
import itertools

matplotlib.style.use('ggplot') #makes plots look pretty

# Generate a dataset and plot it
x, y = sklearn.datasets.make_moons(50, 0, noise=0.1)
xx, yy = sklearn.datasets.make_circles(60, 0, noise=0.08 , factor=0.1)



plt.scatter(xx[:,0], xx[:,1], s=25, c=yy, cmap=plt.cm.Spectral, marker='o' )
plt.scatter(x[:,0], x[:,1], s=50, c=y, cmap=plt.cm.Spectral , marker='x')

# data = np.arra

# kmeans = cl.KMeans(n_clusters=2, random_state=0).fit(x)



plt.show()



