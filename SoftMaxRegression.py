import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('E:/AI/LearningData/iris.csv')
# 准备数据
X = df.drop('Species', axis=1).values[:, 1:]

y = df['Species']
species = np.unique(y)

yConcert = np.zeros((len(y), 1))
for i in np.arange(len(y)):
    yConcert[i] = np.where(species == y[i])[0]
yConcert = yConcert.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, yConcert, test_size=0.4, random_state=50)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)
scaled_X_train = np.hstack((np.ones((len(scaled_X_train), 1)), scaled_X_train))
scaled_X_test = np.hstack((np.ones((len(scaled_X_test), 1)), scaled_X_test))

theta = np.random.random((scaled_X_train.shape[1], len(species)))


def softmax(theta, X):
    return np.exp(np.dot(X, theta)) / np.sum(np.exp(np.dot(X, theta)), axis=1).reshape(X.shape[0], 1)


def cost(theta, X, trainDataValue, arpha, Lamda):
    m = X.shape[0]
    # 非正则化
    # F = (-1 / m) * np.sum(np.dot(trainDataValue.T, np.log(sigmoid(hx(X, theta)))) + np.dot((1 - trainDataValue).T
    #     , np.log(1 - sigmoid(hx(X, theta)))))
    # theta = theta + (-1 / m) * arpha * np.dot(X.T, np.sum(sigmoid(hx(X, theta)) - trainDataValue) * np.ones((m, 1)))
    typeNum = len(np.unique(trainDataValue))
    boolMatrix = trainDataValue == np.arange(typeNum)
    thetaRegular = np.copy(theta)
    thetaRegular[0, :] = 0
    F = -1 / m * np.sum(-1 * np.log(softmax(theta, X)[boolMatrix])) - Lamda / m * np.sum(np.square(theta))
    theta = theta - 1 / m * arpha * (np.dot(X.T, (
            softmax(theta, X) - np.where(boolMatrix, 1, np.zeros((m, typeNum)))))) + arpha * Lamda * thetaRegular / m
    return [F, theta]


def gradientDecent(arpha, Lamda, maxIterationTimes, theta, trainDataFeature, trainDataValue, precise):
    previousF = 0
    iterationTimes = 1
    while True:
        res = cost(theta, trainDataFeature, trainDataValue, arpha, Lamda)
        F = res[0]
        print(f"遍历第 {iterationTimes} 次, 差异值为 {F}")
        # time.sleep(0.01)
        theta = res[1]
        if np.abs(F - previousF) < precise or maxIterationTimes < 0:
            break;
            print("遍历结束")
        previousF = F
        iterationTimes = iterationTimes + 1
        maxIterationTimes = maxIterationTimes - 1
    return theta


arpha = 0.0001
Lamda = 1
maxIterationTimes = 10000000
precise = 0.0000001
# CV = np.array([[True, True, False], [True, True, True]])
# AA = np.zeros((CV.shape))
# AA = np.where(CV, 1, AA)
# print(AA)
# cost(theta, scaled_X_train, y_train, arpha, Lamda)
thetaRes = gradientDecent(arpha, Lamda, maxIterationTimes, theta, scaled_X_train, y_train, precise)

train_res = np.argmax(softmax(thetaRes, scaled_X_train), axis=1)
print(F"训练集预准率为：{np.sum((train_res.reshape(-1, 1) == y_train) != 0) / len(train_res)}")

test_res = np.argmax(softmax(thetaRes, scaled_X_test), axis=1)
print(F"训练集预准率为：{np.sum((test_res.reshape(-1, 1) == y_test) != 0) / len(scaled_X_test)}")
