import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt

plt.style.use('fivethirtyeight')
from sklearn.metrics import classification_report


def logisticRegression():
    data = pd.read_csv('DataFile/ex2data1.txt', header=None, names=['exam1', 'exam2', 'admitted'])
    # print(data.head())
    print(data.describe())
    # 设置图片样式参数，默认主题是灰色背景和白色网格
    sns.set(context='notebook', style='darkgrid', palette=sns.color_palette('RdBu', 2))
    # hue参数是将name所指定的不同类型的数据叠加在一张图片中显示
    sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, height=6, fit_reg=False, scatter_kws={'s': 50})
    plt.show()

    # 分别取出训练的输入样本和输出样本
    X = get_X(data)
    y = get_y(data)

    # 显示sigmoid函数
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(-10, 10, step=0.01), sigmoid(np.arange(-10, 10, step=0.01)))
    ax.set_ylim((-0.1, 1.1))
    ax.set_xlabel('z', fontsize=18)
    ax.set_ylabel('g(z)', fontsize=18)
    ax.set_title('sigmoid function', fontsize=18)
    plt.show()

    theta = np.zeros(3)  # 每一个x都是一个3维的行向量
    print(cost(theta, X, y))
    g = gradient(theta, X, y)
    print(g)

    # 牛顿共轭梯度下降法①
    # lastTheta, lastCost, *unused2 = opt.fmin_ncg(f=cost, fprime=gradient, x0=theta, args=(X, y),
    # maxiter=400,full_output=True)
    # print(lastTheta)
    # 测试
    # x_test = [1, 89, 85]
    # print('predict: ' + str(predict(x_test, lastTheta)))

    # 牛顿共轭梯度下降法②
    res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
    print(res)

    # 获取最优向量解
    finalTheta = res.x

    y_pred = predict(X, finalTheta)  # 预测值
    print(classification_report(y, y_pred))  # 评估

    coef = -(res.x / res.x[2])
    print('coef是：' + str(coef))

    # 计算函数方程
    x = np.arange(130, step=0.01)
    y = coef[0] + coef[1] * x

    # data.describe()
    sns.set(context='notebook', style='ticks', font_scale=1.5)
    sns.lmplot('exam1', 'exam2', hue='admitted', data=data, height=6, fit_reg=False, scatter_kws={'s': 25})
    plt.plot(x, y, 'grey')
    plt.xlim(0, 130)
    plt.ylim(0, 130)
    plt.title('Decision Boundary')
    plt.show()


def predict(x, theta):
    prob = sigmoid(x.dot(theta))
    # 当估计值大于0.5时，返回true，变为int的1
    return (prob >= 0.5).astype(int)


def get_X(df):  # 读取特征
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并 axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并 加列
    # print(data.shape)
    # print(data.head())
    return data.iloc[:, :-1].values  # 这个操作返回 ndarray,不是矩阵,用values将df从数据集变为数组形式
    # 以下形式等价
    # df.insert(0, 'ones', 1)
    # print(df.head())
    # print(df.shape)
    # return df.iloc[:, :-1]


def get_y(df):
    return np.array(df.iloc[:, -1])  # 从数据集形式变为数组形式


def normalize_feature(df):  # 特征提取（数据标准化）
    return df.apply(lambda column: (column - column.mean()) / column.std())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):  # 代价函数J(theta)的计算
    return np.mean(-y * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))


def gradient(theta, X, y):
    return (1 / len(X)) * X.T.dot(sigmoid(X.dot(theta)) - y)
