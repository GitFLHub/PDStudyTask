import os
import time

import numpy as np
import cupy as cp
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

import CommonFunction
import translate as tl
import scipy.io as io


def sigmoid(x):  # 激活函数
    return 1 / (1 + np.exp(-x))


data1 = io.loadmat('E:\\AI\\Machine Learning\\Task\\Task\\Task04\\ex4\\ex4data1')

X = data1["X"]
Y = data1["y"]
pictureSize = 20

X_train = []
# X = []
# Y = []

# dirname = tl.translate
# for dir in dirname:
#     for file in os.open(r'E:\\AI\\LearningData\\Animals\\raw-img\\' + dir):
#         try:
#             print(dir, file)
#             im = Image.open(os.open('E:\\AI\\LearningData\\Animals\\raw-img\\' + dir, file))
#             im = im.resize((64, 64), Image.ANTIALIAS)
#             image_array = np.array(im.convert("L")).flatten()
#             X_train = np.concatenate((X_train, image_array), axis=0)
#         except:
#             pass

# i = 0
# if os.path.exists('.//Data//BP//XSmall.csv') and os.path.exists('.//Data//BP//YSmall.csv'):
#     print("读取 X")
#     X = np.loadtxt('.//Data//BP//XSmall.csv', delimiter=',', dtype=float)
#
#     print("读取 Y")
#     # np.savetxt('.//Data//BP//YSmall.csv', Y, fmt='%s', delimiter=',')
#     Y = np.loadtxt('.//Data//BP//YSmall.csv', delimiter=',', dtype=str)
# if len(Y) == 0 or len(X) == 0:
#     # for dirname, _, filenames in os.walk(r'E:\\AI\\LearningData\\Animals\\raw-img\\'):
#     for dirname, _, filenames in os.walk(r'E:\\AI\\LearningData\\Animals\\TwoClass\\'):
#     # for dirname, _, filenames in os.walk(r'E:\\AI\\LearningData\\Animals\\temp\\'):
#         if len(filenames) != 0:
#             type = os.path.split(dirname)[1]
#         for filename in filenames:
#             # if X_train.shape[0] > 3:
#             #     break
#             try:
#                 print(dirname, filename)
#                 im = Image.open(os.path.join(dirname, filename))
#                 im = im.resize((pictureSize, pictureSize), Image.ANTIALIAS)
#                 im = im.convert("L")
#                 # fig = plt.figure()
#                 # plt.imshow(im)
#                 # plt.show()
#                 image_array = np.array(im)
#                 image_array = image_array.reshape(pictureSize * pictureSize)
#                 X.append(image_array)
#                 Y.append(type)
#                 # X_train = np.concatenate((X_train, image_array), axis=0)
#                 # Y = np.concatenate((Y, type), axis=0)
#             except:
#                 pass
#     X = np.array(X)
#     Y = np.array(Y)
#     print("写入 X")
#     np.savetxt('.//Data//BP//XSmall.csv', X, fmt='%.2f', delimiter=',')
#     print("写入 Y")
#     np.savetxt('.//Data//BP//YSmall.csv', Y, fmt='%s', delimiter=',')

# X = CommonFunction.normalization(X)
multi = CommonFunction.multiType(Y)
Y = multi[0]
ClassType = multi[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)
m = X_train.shape[0]
X_train = cp.asarray(X_train)
X_test = cp.asarray(X_test)
y_train = cp.asarray(y_train)
y_test = cp.asarray(y_test)

# layers = np.array([pictureSize ** 2, pictureSize * 5, pictureSize, len(ClassType)]).astype(int)
layers = np.array([400, 25, 10])
Theta0 = cp.random.random((layers[0] + 1, layers[1]))
Theta1 = cp.random.random((layers[1] + 1, layers[2]))
# Theta2 = cp.random.random((layers[2] + 1, layers[3]))
# Theta3 = cp.random.random((layers[3] + 1, layers[4]))
Lamda = 10
ArphaOri = 0.01
J = 0
JPre = 0
precies = 0.00000001

iterationTimes = 1
previousJ = 0
Theta0Copy = np.copy(Theta0)
Theta1Copy = np.copy(Theta1)
# Theta2Copy = np.copy(Theta2)
while True:
    Theta0Regular = cp.copy(Theta0)
    Theta0Regular[0, :] = 0
    Theta1Regular = cp.copy(Theta1)
    Theta1Regular[0, :] = 0
    # Theta2Regular = cp.copy(Theta2)
    # Theta2Regular[0, :] = 0
    # Theta3Regular = cp.copy(Theta3)
    # Theta3Regular[0, :] = 0

    X0 = np.hstack((cp.ones((m, 1)), X_train))
    Z1 = sigmoid(cp.dot(X0, Theta0))
    X1 = np.hstack((cp.ones((m, 1)), Z1))
    Z2 = sigmoid(cp.dot(X1, Theta1))
    # X2 = np.hstack((cp.ones((m, 1)), Z2))
    # Z3 = sigmoid(cp.dot(X2, Theta2))
    # X3 = np.hstack((cp.ones((m, 1)), Z3))
    # Z4 = sigmoid(cp.dot(X3, Theta3))

    # Arpha = ArphaOri / (1 + iterationTimes * 0.01)
    Arpha = ArphaOri

    # J = - 1 / m * Arpha * (cp.sum(cp.log(Z4) * y_train + (1 - y_train) * cp.log(1 - Z4))) + Lamda / (2 * m) * Arpha * (
    #         cp.sum(Theta0Regular * Theta0Regular) + cp.sum(Theta1Regular * Theta1Regular) + cp.sum(
    #     Theta2Regular * Theta2Regular) + cp.sum(Theta3Regular * Theta3Regular))
    J = 1 / m * (cp.sum(cp.log(Z2) * y_train + (1 - y_train) * cp.log(1 - Z2))) + Lamda / (2 * m) * (
                cp.sum(Theta0Regular * Theta0Regular) + cp.sum(Theta1Regular * Theta1Regular))
    print(f"遍历第 {iterationTimes} 次, 差异值为 {J}")
    if np.abs(J - previousJ) < precies:  # or maxIterationTimes < 0:
        break
        print("遍历结束")
    previousJ = J
    iterationTimes = iterationTimes + 1

    # sigma3 = (Z3 - y_train).T
    sigma2 = (y_train - Z2).T

    # sigma4 = (y_train - Z4).T
    # sigma4 = (Z4 - y_train).T
    # sigma3 = cp.dot(Theta3[1:], sigma4) * sigmoid(X3[:, 1:].T)
    # sigma2 = cp.dot(Theta2[1:], sigma3) * sigmoid(X2[:, 1:].T)
    sigma1 = cp.dot(Theta1[1:], sigma2) * sigmoid(X1[:, 1:].T)
    sigma0 = cp.dot(Theta0[1:], sigma1) * sigmoid(X0[:, 1:].T)
    # Theta3 = Theta3 + Arpha / m * cp.dot(sigma4, X3).T - Arpha * Lamda / m * Theta3
    # Theta2 = Theta2 + Arpha / m * cp.dot(sigma3, X2).T - Lamda / m * Theta2
    Theta1 = Theta1 + Arpha / m * cp.dot(sigma2, X1).T - Lamda / m * Theta1
    Theta0 = Theta0 + Arpha / m * cp.dot(sigma1, X0).T - Lamda / m * Theta0

print("训练结束")
Theta0 - Theta0Copy
Theta1 - Theta1Copy
# Theta2 - Theta2Copy
mT = X_test.shape[0]
X_test = np.hstack((np.ones((mT, 1)), X_test))

X0 = np.hstack((np.ones((m, 1)), X_train))
Z1 = sigmoid(cp.dot(X0, Theta0))
X1 = np.hstack((np.ones((m, 1)), Z1))
Z2 = sigmoid(cp.dot(X1, Theta1))
# X2 = np.hstack((np.ones((m, 1)), Z2))
# Z3 = sigmoid(cp.dot(X2, Theta2))
# X3 = np.hstack((np.ones((mT, 1)), Z3))
# Z4 = sigmoid(cp.dot(X3, Theta3))
predictT = []
print(np.unique(np.argmax(Z2, axis=1)))
print(np.argmax(Z2, axis=1))
# predictIndex = np.argmax(Z3, axis=1)
# for i in np.arange(m):
#     predictT = ClassType[predictIndex[i]]

# for i in np.arange(m):
#     X_test_imgArray = X_train[i]
#
#     X0p = np.hstack((1, X_train[i]))
#     Z1p = sigmoid(cp.dot(X0p, Theta0))
#     X1p = np.hstack((1, Z1p))
#     Z2p = sigmoid(cp.dot(X1p, Theta1))
#     X2p = np.hstack((1, Z2p))
#     Z3p = sigmoid(cp.dot(X2p, Theta2))
#     # X3 = np.hstack((1, Z3))
#     # Z4 = sigmoid(cp.dot(X3, Theta3))
#     # predictT = tl.translate[ClassType[str(np.argmax(Z3))]]
#     predictT = ClassType[str(np.argmax(Z3p))]
#
#     X_test_imgArray = X_test_imgArray.get()
#     X_test_imgArray = X_test_imgArray / (X_test_imgArray.max() - X_test_imgArray.min()) * 255
#     # fig = plt.figure()
#     # plt.close()
#     plt.imshow(X_test_imgArray.reshape((pictureSize, pictureSize)).T, cmap=plt.cm.gray, vmin=0, vmax=255)
#     plt.axis('off')
#     # plt.title(f"predict: {predictT}, actually: {tl.translate[ClassType[str(np.argmax(y_test[i]))]]}")
#     plt.title(f"predict: {predictT}, actually: {ClassType[str(np.argmax(y_train[i]))]}")
#     # plt.show()
#     # plt.imshow(im)
#     plt.show()

sameCount = 0
for i in np.arange(mT):
    X_test_imgArray = X_test[i, 1:]

    X0p = X_test[i]
    Z1p = sigmoid(cp.dot(X0p, Theta0))
    X1p = np.hstack((1, Z1p))
    Z2p = sigmoid(cp.dot(X1p, Theta1))
    # X2p = np.hstack((1, Z2p))
    # Z3p = sigmoid(cp.dot(X2p, Theta2))
    # X3 = np.hstack((1, Z3))
    # Z4 = sigmoid(cp.dot(X3, Theta3))
    # predictT = tl.translate[ClassType[str(np.argmax(Z3))]]
    predictT = ClassType[str(np.argmax(Z2p))]
    actually = ClassType[str(np.argmax(y_test[i]))]
    if predictT == actually:
        sameCount = sameCount + 1

    # X_test_imgArray = X_test_imgArray.get()
    # X_test_imgArray = X_test_imgArray / (X_test_imgArray.max() - X_test_imgArray.min()) * 255
    # # fig = plt.figure()
    # # plt.close()
    # plt.imshow(X_test_imgArray.reshape((pictureSize, pictureSize)).T, cmap=plt.cm.gray, vmin=0, vmax=255)
    # plt.axis('off')
    # # plt.title(f"predict: {predictT}, actually: {tl.translate[ClassType[str(np.argmax(y_test[i]))]]}")
    # plt.title(f"predict: {predictT}, actually: {actually}")
    # # plt.show()
    # # plt.imshow(im)
    # plt.show()

print(f"测试集准确率 { str(sameCount / mT) }")

