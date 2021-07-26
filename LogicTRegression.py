import pandas as pd
import numpy as np
from sklearn import utils


def split(rawData, splitArr):
    len = rawData.shape[0];
    list = []
    lastEnd = 0
    for i in np.arange(splitArr.shape[0]):
        thisEnd = int(len * splitArr[i])
        list.append(rawData[lastEnd:thisEnd, :])
        lastEnd = thisEnd
    return list


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hx(X, theta):
    return np.dot(X, theta)


def cost(theta, X, trainDataValue, arpha, lamda):
    m = X.shape[0]
    # 非正则化
    # F = (-1 / m) * np.sum(np.dot(trainDataValue.T, np.log(sigmoid(hx(X, theta)))) + np.dot((1 - trainDataValue).T
    #     , np.log(1 - sigmoid(hx(X, theta)))))
    # theta = theta + (-1 / m) * arpha * np.dot(X.T, np.sum(sigmoid(hx(X, theta)) - trainDataValue) * np.ones((m, 1)))
    F = (-0.5 / m) * np.sum(np.square(trainDataValue.T - sigmoid(hx(X, theta))) + lamda * np.sum(np.square(theta)))
    thetaRegular = np.copy(theta)
    thetaRegular[0] = 0
    theta = theta + (-1 / m) * arpha * (np.dot(X.T, np.sum(sigmoid(hx(X, theta)) - trainDataValue) * np.ones(
        (m, 1))) + lamda * thetaRegular)
    return [F, theta]


def normalization(array):
    return (array - array.mean(axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0))
def gradientDecent(arpha, Lambda, maxIterationTimes, theta, trainDataFeature, trainDataValue):
    previousF = 0
    iterationTimes = 1
    arphaT = arpha
    while True:
        res = cost(theta, trainDataFeature, trainDataValue, arphaT, Lambda)
        F = res[0]
        print(f"遍历第 {iterationTimes} 次, 差异值为 {F}")
        theta = res[1]
        if np.abs(F - previousF) < 0.000001 or maxIterationTimes < 0:
            break;
            print("遍历结束")
        arphaT = arpha / (1 + iterationTimes * 0.0000001)
        previousF = F
        iterationTimes = iterationTimes + 1
        maxIterationTimes = maxIterationTimes - 1
    return theta


columnName = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']
data = pd.read_csv("F:\\MachineLearning\\breast-cancer-wisconsin.data", names=columnName)

data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
rawData = (data.values).astype(float)
rawData = utils.shuffle(rawData[:, 1:])

print(rawData.shape)
splitArr = np.array([0.6, 1])
splitData = split(rawData, splitArr)
trainData = splitData[0]
testData = splitData[1]
trainSplit = np.hsplit(trainData, [trainData.shape[1] - 1])
testSplit = np.hsplit(testData, [testData.shape[1] - 1])

trainDataFeature = np.insert(normalization(trainSplit[0]), 0, values=np.ones((trainSplit[0].shape[0])), axis=1)
trainDataValue = (trainSplit[1] - 2) / 2

testDataFeature = np.insert(normalization(testSplit[0]), 0, values=np.ones((testSplit[0].shape[0])), axis=1)
testDataValue = (testSplit[1] - 2) / 2

theta = np.random.random((trainDataFeature.shape[1], 1))

arpha = 0.01
Lambda = 10
maxIterationTimes = 100000000
theta = gradientDecent(arpha, Lambda, maxIterationTimes, theta, trainDataFeature, trainDataValue)


res = sigmoid(hx(testDataFeature, theta))
res[res > 0.5] = 1
res[res <= 0.5] = 0
print(np.hstack((res, testDataValue, res - testDataValue)))
print(f"测试集预准率为{1 - np.sum(np.abs(res - testDataValue)) / res.shape[0]}")

resTrain = sigmoid(hx(trainDataFeature, theta))
resTrain[resTrain > 0.5] = 1
resTrain[resTrain <= 0.5] = 0
print(np.hstack((resTrain, trainDataValue, resTrain - trainDataValue)))
print(f"训练集预准率为{1 - np.sum(np.abs(resTrain - trainDataValue)) / resTrain.shape[0]}")
