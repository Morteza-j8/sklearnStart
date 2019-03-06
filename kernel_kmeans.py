# Load the data
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cluster as cl
import numpy as np
import random

data, label = sklearn.datasets.make_circles(500, 0, noise=0.15, factor=0.2)

K = np.zeros((len(data) , len(data)))



for i in range(len(data)):
    for j in range(len(data)):
        K[i][j] = np.linalg.norm(data[i] - data[j])


real_labels = [0, 1]

t = 0

cur_label = []


cur_label.append([])
for i in range(len(data)):
    cur_label[t].append(random.choice(real_labels))


while True:
    t = t + 1

    sqnorm = []
    for i in range(len(real_labels)):
        cur_sq = 0
        n = 0
        for r in range(len(K)):
            for c in range(len(K[r])):
                if cur_label[t-1][r] == cur_label[t-1][c] and cur_label[t-1][r] == real_labels[i]:
                    cur_sq += K[r][c]
                    n += 1
        if n > 0:
            cur_sq /= (n*n)

        sqnorm[i] = cur_sq






plt.scatter(data[:,0], data[:,1], s = 30 , c = cur_label, cmap=plt.cm.Spectral)
plt.show()






#
# kmeans = cl.KMeans(n_clusters=2, random_state=0).fit(two_moon_x)
# labels = kmeans.labels_
# predicate = kmeans.predict([[0, 0], [12, 3]])
# cl_center = kmeans.cluster_centers_
#
# matplotlib.style.use('ggplot') #makes plots look pretty
# plt.scatter(two_moon_x[:,0],two_moon_x[:,1], s = 30 , c = labels, cmap=plt.cm.Spectral)
# plt.show()




