import numpy as np
import pandas as pd
import CommonFunction as cf
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as io

data1 = io.loadmat('E:\\StudySpace\\AI\\Machine Learning\\Task\\Task07\\ex7\\ex7data1')
data2 = io.loadmat('E:\\StudySpace\\AI\\Machine Learning\\Task\\Task07\\ex7\\ex7data2')
data3 = io.loadmat('E:\\StudySpace\\AI\\Machine Learning\\Task\\Task07\\ex7\\ex7faces')

mat = np.array([[-1, 1, 1],
                [2, -1, 1],
                [1, 0, 2]])
X = cf.normalization(data2['X'])
# X = cf.normalization(mat)
ax = plt.axes(projection='3d')  # 设置三维轴
X = X[:100, ]
# ax.scatter3D(X[:, 0], X[:, 1], X[:, 2])  # 三个数组对应三个维度（三个数组中的数一一对应）
plt.scatter(X[:, 0], X[:, 1]) # , X[:, 2]
m = X.shape[0]
sigma = 1 / m * np.dot(X, X.T)
# temp = np.array([[1, 2],[2,4]])

eigenvalue, featureVector = np.linalg.eig(sigma)
# eigenvalue, featureVector = np.linalg.eig(temp)
S = np.diag(eigenvalue)
u = 0
for i in np.arange(len(eigenvalue)):
    if np.sum(eigenvalue[0:i]) / np.sum(eigenvalue) >= 0.99:
        u = i
        break
z = np.dot(featureVector[:, 0:u].T, X)
# ax.scatter3D(z[:, 0], z[:, 1], z[:, 2])  # 三个数组对应三个维度（三个数组中的数一一对应）
plt.scatter(z[:, 0], z[:, 1])  # , z[:, 2]

# plt.scatter(z, z)
plt.show()
input()
