import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import threadpool


def split(rawData, splitArr):
    len = rawData.shape[0]
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


def cost1(theta, X, trainDataValue, arpha, lamda):
    m = X.shape[0]
    # 非正则化
    # F = (-1 / m) * np.sum(np.dot(trainDataValue.T, np.log(sigmoid(hx(X, theta)))) + np.dot((1 - trainDataValue).T
    #     , np.log(1 - sigmoid(hx(X, theta)))))
    # theta = theta + (-1 / m) * arpha * np.dot(X.T, np.sum(sigmoid(hx(X, theta)) - trainDataValue) * np.ones((m, 1)))
    F = (-0.5 / m) * np.sum(np.square(trainDataValue.T - sigmoid(hx(X, theta))) + lamda * np.sum(np.square(theta)))
    thetaRegular = np.copy(theta)
    thetaRegular[0] = 0
    theta = theta + (-1 / m) * arpha * (np.dot(X.T, np.sum(sigmoid(hx(X, theta)) - trainDataValue) * np.ones(
        (m, 1))) + lamda * thetaRegular / m)
    return [F, theta]


def normalization(array):
    return (array - array.mean(axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0))


def normalization(array):
    return (array - array.mean(axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0))


def get_result(request, result):
    global thetaRes
    thetaRes[:, [result[0]]] = result[1]


def decentThread(arpha, Lambda, maxIterationTimes, theta, scaled_X_train, y_train, precies, i):
    print(F"训练 Theta {i}\n")
    y_train_normalize = np.zeros(y_train.shape)
    y_train_normalize[y_train == species[i]] = 1
    y_train_normalize[y_train != species[i]] = 0
    # theta[:, [i]] = gradientDecent(arpha, Lambda, maxIterationTimes, theta[:, [i]], scaled_X_train, y_train_normalize,
    #                                precies, i)
    return [i, gradientDecent(arpha, Lambda, maxIterationTimes, theta[:, [i]], scaled_X_train, y_train_normalize,
                              precies, i)]
    # return theta[:, [i]]


def gradientDecent1(arpha, Lambda, maxIterationTimes, theta, trainDataFeature, trainDataValue, precies, i):
    previousF = 0
    iterationTimes = 1
    arphaT = arpha
    while True:
        res = cost(theta, trainDataFeature, trainDataValue, arphaT, Lambda)
        F = res[0]
        theta = res[1]
        tab = '\t' * i * 15
        if (F - previousF) >= 0:
            print(f"\033[32m{tab}遍历第 {iterationTimes} 次, 差异值为 {F}\n")
        else:
            print(f"\033[31m{tab}遍历第 {iterationTimes} 次, 差异值为 {F}\n")
        if np.abs(F - previousF) < precies or maxIterationTimes < 0:
            print(f"{tab}遍历结束")
            break
        arphaT = arpha / (1 + iterationTimes * 0.1)
        previousF = F
        iterationTimes = iterationTimes + 1
        maxIterationTimes = maxIterationTimes - 1
    return theta

def probabilityFun(X, theta):
    return np.exp(np.dot(X, theta))
def gradientDecent(arpha, Lambda, maxIterationTimes, theta, trainDataFeature, trainDataValue, precies):
    previousF = 0
    iterationTimes = 1
    arphaT = arpha
    while True:
        res = cost(theta, trainDataFeature, trainDataValue, arphaT, Lambda)
        F = res[0]
        theta = res[1]
        tab = '\t' * i * 15
        if (F - previousF) >= 0:
            print(f"\033[32m{tab}遍历第 {iterationTimes} 次, 差异值为 {F}\n")
        else:
            print(f"\033[31m{tab}遍历第 {iterationTimes} 次, 差异值为 {F}\n")
        if np.abs(F - previousF) < precies or maxIterationTimes < 0:
            print(f"{tab}遍历结束")
            break
        arphaT = arpha / (1 + iterationTimes * 0.1)
        previousF = F
        iterationTimes = iterationTimes + 1
        maxIterationTimes = maxIterationTimes - 1
    return theta

def cost(theta, X, trainDataValue, arpha, lamda):
    m = X.shape[0]
    # 非正则化
    # F = (-1 / m) * np.sum(np.dot(trainDataValue.T, np.log(sigmoid(hx(X, theta)))) + np.dot((1 - trainDataValue).T
    #     , np.log(1 - sigmoid(hx(X, theta)))))
    # theta = theta + (-1 / m) * arpha * np.dot(X.T, np.sum(sigmoid(hx(X, theta)) - trainDataValue) * np.ones((m, 1)))
    # F = (-0.5 / m) * np.sum(np.square(trainDataValue.T - sigmoid(hx(X, theta))) + lamda * np.sum(np.square(theta)))
    # thetaRegular = np.copy(theta)
    # thetaRegular[0] = 0
    # theta = theta + (-1 / m) * arpha * (np.dot(X.T, np.sum(sigmoid(hx(X, theta)) - trainDataValue) * np.ones(
    #     (m, 1))) + lamda * thetaRegular / m)
    F = -1 /m
    return [F, theta]


df = pd.read_csv('E:/AI/LearningData/iris.csv')
# 准备数据
X = df.drop('Species', axis=1).values[:, 1:]
y = df['Species']
species = np.unique(y)

yConcert = np.zeros((len(y), 1))
for i in np.arange(len(y)):
    yConcert[i] = np.where(species == y[i])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
scaler = StandardScaler()
# scaled_X_train = scaler.fit_transform(X_train)
# scaled_X_test = scaler.fit_transform(X_test)
scaled_X_train = normalization(X_train)
scaled_X_test = normalization(X_test)



# for i in np.arange(len(species)):
#     print(F"训练 Theta {i}")
#     y_train_normalize = np.zeros(y_train.shape)
#     y_train_normalize[y_train == species[i]] = 1
#     y_train_normalize[y_train != species[i]] = 0
#     theta[:, [i]] = gradientDecent(arpha, Lambda, maxIterationTimes, theta[:, [i]], scaled_X_train, y_train_normalize,
#                                    precies, i)

theta = np.random.random((scaled_X_train.shape[1], species.shape[0]))
# theta = np.ones((scaled_X_train.shape[1], species.shape[0]))
thetaRes = np.zeros((scaled_X_train.shape[1], species.shape[0]))

arpha = 30000
Lambda = 1
maxIterationTimes = 100000000
precies = 0.00000001
dictVars1 = {"arpha": arpha, 'Lambda': Lambda, 'maxIterationTimes': maxIterationTimes, 'theta': theta,
             'scaled_X_train': scaled_X_train, 'y_train': y_train, 'precies': precies, 'i': 0}
dictVars2 = {"arpha": arpha, 'Lambda': Lambda, 'maxIterationTimes': maxIterationTimes, 'theta': theta,
             'scaled_X_train': scaled_X_train, 'y_train': y_train, 'precies': precies, 'i': 1}
dictVars3 = {"arpha": arpha, 'Lambda': Lambda, 'maxIterationTimes': maxIterationTimes, 'theta': theta,
             'scaled_X_train': scaled_X_train, 'y_train': y_train, 'precies': precies, 'i': 2}
funcVar = [(None, dictVars1), (None, dictVars2), (None, dictVars3)]
# 声明可容纳五个线程的池
pool = threadpool.ThreadPool(3)
# 创建线程运行内容请求列表（线程工作函数，线程工作参数列表，回调函数）
requests = threadpool.makeRequests(decentThread, funcVar, get_result)
# 将每一个线程请求扔进线程池
[pool.putRequest(req) for req in requests]
# 等待data被消耗完，所有线程运行结束。
pool.wait()
print(theta)
print(thetaRes)

train_res = species[np.argmax(np.exp(hx(scaled_X_train, thetaRes)), axis=1)]
y_train_res = np.asarray(y_train)
res = species[np.argmax(sigmoid(hx(scaled_X_test, thetaRes)), axis=1)]
y_test_res = np.asarray(y_test)
# for i in np.unique(y_train):
#     print(f"{i} 训练集占比 {np.sum(y_train == i) / len(y_train)} 测试集占比 {np.sum(y_test == i) / len(y_test)}")

print(F"训练集预准率为：{np.sum((train_res == y_train_res) != 0) / len(train_res)}")
print(F"测试集预准率为：{np.sum((res == y_test_res) != 0) / len(res)}")
