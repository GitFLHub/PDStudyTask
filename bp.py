import os
from sklearn.neural_network import MLPRegressor

import numpy as np

import os

from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import CommonFunction
import translate as tl


def multiType(array):
    type = np.unique(array)
    m = array.shape[0]
    res = np.zeros((m, 1))
    dict = {}
    index = np.arange((len(type)))
    for i in np.arange(m):
        res[i, 0] = index[np.where(type == array[i])]
    for j in np.arange(len(type)):
        dict[str(j)] = type[j]
    return [res, dict]


pictureSize = 64
X = []
Y = []
if os.path.exists('.//Data//BP//X.csv') and os.path.exists('.//Data//BP//Y.csv'):
    print("读取 X")
    X = np.loadtxt('.//Data//BP//X.csv', delimiter=',', dtype=float)

    print("读取 Y")
    # np.savetxt('.//Data//BP//Y.csv', Y, fmt='%s', delimiter=',')
    Y = np.loadtxt('.//Data//BP//Y.csv', delimiter=',', dtype=str)
if len(Y) == 0 or len(X) == 0:
    for dirname, _, filenames in os.walk(r'E:\\AI\\LearningData\\Animals\\raw-img\\'):
        # for dirname, _, filenames in os.walk(r'E:\\AI\\LearningData\\Animals\\TwoClass\\'):
        if len(filenames) != 0:
            type = os.path.split(dirname)[1]
        for filename in filenames:
            # if X_train.shape[0] > 3:
            #     break
            try:
                print(dirname, filename)
                im = Image.open(os.path.join(dirname, filename))
                im = im.resize((pictureSize, pictureSize), Image.ANTIALIAS)
                im = im.convert("L")
                # fig = plt.figure()
                # plt.imshow(im)
                # plt.show()
                image_array = np.array(im)
                image_array = image_array.reshape(pictureSize * pictureSize)
                X.append(image_array)
                Y.append(type)
                # X_train = np.concatenate((X_train, image_array), axis=0)
                # Y = np.concatenate((Y, type), axis=0)
            except:
                pass
    X = np.array(X)
    Y = np.array(Y)
    print("写入 X")
    np.savetxt('.//Data//BP//X.csv', X, fmt='%.2f', delimiter=',')
    print("写入 Y")
    np.savetxt('.//Data//BP//Y.csv', Y, fmt='%s', delimiter=',')

X = CommonFunction.normalization(X)
multi = multiType(Y)
Y = multi[0]
ClassType = multi[1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)
m = X_train.shape[0]
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
# data_tr = np.hstack((X_train, y_train))
# data_te = np.hstack((X_test, y_test))
model = MLPRegressor(hidden_layer_sizes=(4096, 128, 16, 4, 2), random_state=10, learning_rate_init=0.1)  # BP神经网络回归模型
model.fit(X_train, y_train)  # 训练模型
pre = model.predict(X_test)  # 模型预测
np.abs(y_test - pre).mean()  # 模型评价

mT = X_test.shape[0]
X_test = np.hstack((np.ones((mT, 1)), X_test))
for i in np.arange(mT):
    X_test_imgArray = X_test[i, 1:]
    predictT = tl.translate[ClassType[str(np.argmax(pre[i]))]]

    X_test_imgArray = X_test_imgArray / (X_test_imgArray.max() - X_test_imgArray.min()) * 255
    # fig = plt.figure()
    # plt.close()
    plt.imshow(X_test_imgArray.reshape((pictureSize, pictureSize)), cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.axis('off')
    plt.title(f"predict: {predictT}, actually: {tl.translate[ClassType[str(np.argmax(y_test[i]))]]}")
    # plt.show()
    # plt.imshow(im)
    plt.show()
