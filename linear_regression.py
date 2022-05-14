
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def LinearRegression_one():
    # 引入数据文件
    path = 'DataFile/ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    print(data.head()) #预览数据
    # print(data.describe())

    # 绘制原始数据散点图
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    plt.show()
    data.insert(0, 'Ones', 1)  # 在训练集中再添加一列虚拟参数，以便我们可以使用向量化的解决方案来计算代价和梯度
    cols = data.shape[1]  # 获取现在的训练集的列数
    X = data.iloc[:, 0:cols - 1]  # 获取所有行，第0列和第1列,也就是输入量
    y = data.iloc[:, cols - 1:cols]  # 获取输出量

    # #打印输入集X和输出集y
    # print(X.head())
    # print(y.head())
    # 将训练输入集和训练输出集都矩阵化
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # 给参数theta矩阵化
    theta = np.matrix(np.array([0, 0]))
    # 查看X,y,theta的维度
    # print(X.shape, theta.shape, y.shape)
    print('初始的代价函数：' + str(computerCost(X, y, theta)))

    alpha = 0.01
    iters = 1000
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    print('theta 最后的值是: ' + str(g))
    print('梯度下降后的代价函数：' + str(computerCost(X, y, g)))

    x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 抽100个样本
    # 写出线性回归函数，并训练
    f = g[0, 0] + (g[0, 1] * x)  # g[0, 0] 代表theta0 ,g[0, 1]代表theta1

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Training Data')
    ax.legend(loc=4)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


# 多元回归
def LinearRegression_multiple():
    path = 'DataFile/ex1data2.txt'
    data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print(data2.head())  # 显示data2的数据信息
    data2 = (data2 - data2.mean()) / data2.std()
    # print(data2.head())
    # 添加一个x1
    data2.insert(0, 'Ones', 1)
    cols = data2.shape[1]
    X2 = data2.iloc[:, 0:cols - 1]  # 第一个':'表示所有行，后面的0:cols-1表示从第0列到cols-2列，左闭右开区间
    y2 = data2.iloc[:, cols - 1:cols]

    X2 = np.matrix(X2.values)
    y2 = np.matrix(y2.values)
    theta2 = np.matrix(np.array([0, 0, 0]))
    alpha = 0.01
    iters = 1000
    g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

    print('theta 最后的值是: ' + str(g2))
    print('梯度下降后的代价函数：' + str(computerCost(X2, y2, g2)))


    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Training Epoch')
    plt.show()


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])  # ravel计算需要求解的参数个数 概念将多维数组降至一维
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])  # 计算(h(xi)-yi)xi
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))  # 更新theta

        theta = temp
        # 记录每次下降完之后的cost
        cost[i] = computerCost(X, y, theta)
    return theta, cost


def computerCost(X, y, theta):
    # 最小二乘法
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
