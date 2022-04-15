# -*- coding: utf-8 -*-
# ！/usr/bin/python3.9.7
# @Time   : 2022.4.18
# @Author : Yue Gaofeng
# @version: V2.0
# @Des    : 

from time import *
import numpy as np
import matplotlib.pyplot as plt

loss_gru = np.loadtxt("loss_gru.txt")
loss_lstm = np.loadtxt("loss_lstm.txt")
loss_gru = loss_gru.astype(np.float64)
loss_lstm = loss_lstm.astype(np.float64)
two_loss = np.vstack((loss_gru, loss_lstm))
print(two_loss)
plt.figure()
plt.xlabel("Iteration",fontsize=14)
plt.ylabel("Loss (CrossEntropyLoss)",fontsize=14)
plt.title("MNIST Learning curve for LSTM and GRU",fontsize=14)
plt.plot(two_loss.T[:,0],label="GRU")
plt.plot(two_loss.T[:,1], label="LSTM")
plt.legend(fontsize=14)
plt.show()




