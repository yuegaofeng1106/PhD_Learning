# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2022.4.10
# @Author : Yue Gaofeng
# @version: V1.0
# @Des    : Learning DBSCAN

# 导入numpy模块
from tkinter import font
import numpy as np
# 导入sklearn中的cluster模块
from sklearn import cluster
# 导入数据生成模块
from sklearn.datasets import _samples_generator
# 导入matplotlib模块的pyplot类
import matplotlib.pyplot as plt
# 随机生成聚类数据集
def loadDataSet():
    x = []
    f = open('data.csv') # 打开文本文件
    for line in f.readlines():  # 按行迭代读取数据
        lineList = line.strip().split(',') # 按默认字符（空格）拆分数据
        x.append([float(lineList[0]),float(lineList[1])])
    return np.array(x)
X = loadDataSet()
# 定义一个Agglomerative聚类器对象，设置类别数量为4
ac = cluster.AgglomerativeClustering (n_clusters=4, linkage='ward')
# 调用该对象的聚类方法
ac.fit(X)
# 准备绘制聚类结果
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.tick_params(labelsize=14)
plt.title('DBSCAN聚类',fontsize=18)
plt.xlabel('时间序列',fontsize=18)
plt.ylabel('传感器特征值',fontsize=18)

colors=['r','g','b','c']
markers=['o','s','D','+']
# 为聚类结果中的每个类别使用一种颜色和形状绘制散点图
for i, j in enumerate(ac.labels_):
    plt.scatter(X[i][0], X[i][1], color=colors[j], marker=markers[j], s=5)
#plt.legend(['传感器1','传感器2','传感器3','传感器4'],\
#                       fontsize = 18, loc='best')
plt.show()
