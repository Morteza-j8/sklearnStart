# Load the data
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import math


def distance(d1, d2):
    dx = d1[0] - d2[0]
    dy = d1[1] - d2[0]
    return math.sqrt(dx * dx + dy * dy)


EPSILON = 10

data, label = sklearn.datasets.make_circles(500, 0, noise=0.15, factor=0.1)

K = np.zeros((len(data), len(data)))

for i in range(len(data)):
    for j in range(len(data)):
        K[i][j] = distance(data[i], data[j])

real_labels = [0, 1]

t = 0

cur_label = [[]]

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
                if cur_label[t - 1][r] == real_labels[i] and cur_label[t - 1][c] == real_labels[i]:
                    cur_sq += K[r][c]
                    n += 1
        if n > 0:
            cur_sq /= n * n

        sqnorm.append(cur_sq)

    avg = []
    for i in range(len(data)):
        avg.append([])
        for j in range(len(real_labels)):
            y = 0
            n = 0
            for k in range(len(data)):
                if real_labels[j] == cur_label[t - 1][k]:
                    y += K[i][k]
                    n += 1

            if n > 0:
                y /= n

            avg[i].append(y)

    d = []
    for j in range(len(data)):
        d.append([])
        for i in range(len(real_labels)):
            d[j].append(sqnorm[i] - 2 * avg[j][i])

    cur_label.append([])
    for j in range(len(data)):
        min_index = -1
        for k in range(0, len(real_labels)):
            if min_index == -1 or d[j][k] < d[j][min_index]:
                min_index = k

        cur_label[t].append(min_index)

    flag = 0
    for i in range(len(data)):
        if cur_label[t][i] == cur_label[t - 1][i]:
            flag += 1

    if flag - len(data) <= EPSILON:
        break

print("repeat time:", t)

plt.scatter(data[:, 0], data[:, 1], s=60, c=cur_label[t], cmap=plt.cm.Spectral)
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
