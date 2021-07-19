from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np

data_tr = pd.read_csv(r'./data/BPdata_tr.txt')  # 导入数据
data_te = pd.read_csv(r'./data/BPdata_te.txt')
model = MLPRegressor(hidden_layer_sizes=(10,), random_state=10, learning_rate_init=0.1)  # BP神经网络回归模型
model.fit(data_tr.iloc[:, :2], data_tr.iloc[:, 2])  # 训练模型
pre = model.predict(data_te.iloc[:, :2])  # 模型预测
np.abs(data_te.iloc[:, 2] - pre).mean()  # 模型评价
