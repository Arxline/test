import pandas as pd

# df = pd.DataFrame([
#     ['green', 'M', 10.1, 'class1'],
#     ['red', 'L', 13.5, 'class2'],
#     ['blue', 'XL', 15.3, 'class1']], index=range(0, 3), columns=list('ABCD'))
# print('现在的数据集是：')
# print(df)
# print('当前数据集为：' + str(df.shape))
# cols = df.shape[1]  # shape[0]表行数，shape[1]表示列数
# print('列数是：' + str(cols))
#
# print('loc[row,col]通过标签提取数据:')
# print('取出行标签为0的那一行：')
# print(df.loc[0])
# # 也可以用print(df.loc[0,:])
#
# print('取出列标签为B的那一列（’:‘表示全部）：')
# print(df.loc[:, 'B'])
#
# print('取出列标签为B的第0行和第1行(loc左闭右闭)：')
# print(df.loc[0:1, 'B'])
#
# print('iloc[row,col]通过行列坐标来取数据：')
# print('取出第0行：')
# print(df.iloc[0])
# # 也可以用print(df.iloc[0,:])
#
# print('取出第1列(注意：iloc左开右闭)：')
# print(df.iloc[:, 1:2])
#
# print('取出第1行和第二行的后两列：')
# print(df.iloc[1:3, cols - 2:cols])