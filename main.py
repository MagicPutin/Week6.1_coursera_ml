import skimage
import pylab
from sklearn.cluster import KMeans
import numpy as np
# data preprocessing


image = pylab.imread('Data/parrots.jpg')
float_image = skimage.img_as_float(image)

# 2nd step
X = []
for i in float_image:
    for j in i:
        X.append(np.ndarray.tolist(j))

# 3rd step
# color_median[clust] = np.median(objects_features_matrix[np.where(labels == clust_num)], axis=0)

cluster = KMeans(init='k-means++', random_state=241).fit(X)
length = 337962  # len(cluster.labels_) it's so slow, so i commend it
B = [[] for y in range(cluster.n_clusters)]
for i in range(length):
    B[cluster.labels_[i]].append(X[i])

color_mean = [[] for y in range(cluster.n_clusters)]
color_median = [[] for y in range(cluster.n_clusters)]

for i in range(cluster.n_clusters):
    color_mean[i] = np.ndarray.tolist(np.mean(B[i], axis=0))
    color_median[i] = np.ndarray.tolist(np.median(B[i], axis=0))

# 4th step
MSE_mean = 0
for i in range(length):
    for j in range(3):
        MSE_mean += (X[i][j] - color_mean[cluster.labels_[i]][j])**2
MSE_mean /= length
MSE_mean /= 3
PSNR_mean = 10 * np.log10(1/MSE_mean)
print(PSNR_mean)

MSE_median = 0
for i in range(length):
    for j in range(3):
        MSE_median += (X[i][j] - color_median[cluster.labels_[i]][j])**2
MSE_median /= length
PSNR_median = 10 * np.log10(1/MSE_median)
print(PSNR_median)

# 5th step
flag = 0
for clust_num in range(1, 21):
    cluster = KMeans(init='k-means++', random_state=241, n_clusters=clust_num).fit(X)
    B = [[] for y in range(cluster.n_clusters)]
    for i in range(length):
        B[cluster.labels_[i]].append(X[i])

    color_mean = [[] for y in range(cluster.n_clusters)]
    color_median = [[] for y in range(cluster.n_clusters)]

    for i in range(cluster.n_clusters):
        color_mean[i] = np.ndarray.tolist(np.mean(B[i], axis=0))
        color_median[i] = np.ndarray.tolist(np.median(B[i], axis=0))

    # 4th step
    MSE_mean = 0
    for i in range(length):
        for j in range(3):
            MSE_mean += (X[i][j] - color_mean[cluster.labels_[i]][j]) ** 2
    MSE_mean /= length
    MSE_mean /= 3
    PSNR_mean = 10 * np.log10(1 / MSE_mean)
    print(clust_num, end=' ')
    print(PSNR_mean, end=' ')

    MSE_median = 0
    for i in range(length):
        for j in range(3):
            MSE_median += (X[i][j] - color_median[cluster.labels_[i]][j]) ** 2
    MSE_median /= length
    PSNR_median = 10 * np.log10(1 / MSE_median)
    print(PSNR_median)
    if flag != 1 and (PSNR_mean > 20 or PSNR_median > 20) :
        flag = 1
        with open('Answer/task1.txt', 'w') as ans:
            ans.write(str(clust_num))