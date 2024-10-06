# -*- coding: utf-8 -*-
# ！~/anaconda3/bin/python3
# @Time   : 2021.12.4
# @Author : Coly
# @version: V3.0
# @school: X_JT_U
# @Des    : 这个程序主要是为了双层可信联邦学习所写，更新了训练集重复使用的问题。
#            加快速度，非常明显
#            添加了GPU加速

import findspark
findspark.init()
from pyspark.sql import SparkSession
import os
import numpy as np
import syft as sy
import torch
from torch import optim, nn
import torch.nn.functional as F
import math
import datetime
import collections
from torch.autograd import Variable

os.environ['JAVA_HOME'] = '/home/yue_gf/jdk'
os.environ["PYSPARK_PYTHON"] = "/home/yue_gf/anaconda3/bin/python3"


def data_distribution(cs_data):                 #数据分发给客户端
    EV_ID_cnt = np.array(cs_data.groupby('EV_ID').count().collect())
    acc_tar = tamp = 0
    datasets = []
    for index in EV_ID_cnt[:, 0]:
        data_tmp = np.array(cs_data.where(cs_data.EV_ID == index).collect()) #选择一个车
        data_tmp = data_tmp[:,2:4].astype(float)   #原始数据是字符串格式
        input, tar = data_tmp[:,0].tolist(), [data_tmp[0,1]]      #输入和输出
        if max_size > len(input):
             for m in range(0, max_size-len(input)):input+=[input[m]]  #输入向量必须大小一致
        else: input = input[0:max_size]
        #print(input, tar) #调试数据用
        acc_tar += data_tmp[0,1]
        VirtualClient = sy.VirtualWorker(hook, id = index)  #建立联邦学习的客户端
        if(use_gpu):
            data_input = torch.tensor(input, requires_grad=True, dtype=torch.float).cuda().send(VirtualClient) #分发数据给客户端
            data_tar = torch.tensor(tar, requires_grad=True, dtype=torch.float).cuda().send(VirtualClient)
        else:
           data_input = torch.tensor(input, requires_grad=True, dtype=torch.float).send(VirtualClient) #分发数据给客户端
           data_tar = torch.tensor(tar, requires_grad=True, dtype=torch.float).send(VirtualClient)    
        if tamp == 0:datasets, tamp = [(data_input, data_tar)], 1
        else:datasets.append((data_input, data_tar))
    return datasets,acc_tar

class Model(nn.Module):   #定义所需要的网络模型
    def __init__(self):
        super(Model, self).__init__()
        self.w1 = nn.Linear(max_size, 4)
        self.w2 = nn.Linear(4, 3)
        self.w3 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.w1(x))
        x = torch.relu(self.w2(x))
        x = torch.relu(self.w3(x))
        return x

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x, hidden):
        x = x.view(-1, x.size(1)) 
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden) 
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy
       
class GRU(nn.Module):   #定义所需要的网络模型
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)) 
        outs = []
        hn = h0[0,:,:]       
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out) 
        # out.size() --> 100, 10
        return out

def fl_wgt_avg(models, model):  #第二层联邦学习, 加权聚合
    worker_state_dict = [x.state_dict() for x in models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum/len(models)
    # update fed weights to fl model
    model.load_state_dict(fed_state_dict)
    return model  #fanhui

def nn_train(EV_data, cs_cnt):   #双层可信联邦学习
    model = Model() #模型实例化
    #model = GRU(input_dim=3, hidden_dim=3, layer_dim=2, output_dim=1) #模型实例化
    if(use_gpu):  #是否使用GPU训练
        fl_model = model.cuda() #加权准备
        loss_fn = nn.MSELoss().cuda  #设置损失函数
    else:
        fl_model = model 
        loss_fn = nn.MSELoss()     
    opt = optim.SGD(params=model.parameters(), lr=0.0001) #设定模型参数
    list_number = [0]    #为了重复数据集
    models = []
    iter_acc = np.zeros([100,98])  #为了绘制准确度
    acc_tar_sum = acc_pred_sum = 0
    print('The trainning is starting!')  
    for iter in range(100):     #设定迭代次数
        print('The trainning is runing, iter'+str(iter+1)+' processing')
        for cs in range(cs_cnt): #可以控制双层的客户机数量
            if iter == 0:          #这个非常重要  要不然每次都重新建立工人，太费时间了
                print('Sending the '+str(cs+1)+'th CS data')
                cs_data = EV_data.where(EV_data.cs_label == cs)  #选择一个充电站
                datasets, acc_tar_sum = data_distribution(cs_data)        #数据分发给客户端,返回充电站一个的模型，一步的结果
                cnt_tmp = sum(list_number) +  len(datasets)      
                list_number = list_number + [cnt_tmp]   #记录数据集大小，方便调用
                EV_data = EV_data.where(EV_data.cs_label > cs)     #用过的数据丢弃，加快处理
                if cs == 0:
                    cs2ev_datasets = datasets      #为了拼接大数据集,设置中间变量
                else:  #备份首次运行的工人集合
                    cs2ev_datasets = cs2ev_datasets + datasets
            else:
                start, end = list_number[cs], list_number[cs+1]
                datasets = cs2ev_datasets[start:end]   #第二次不需要重新遍历所有数据，用第一次的结果
                model = fl_model     #第二次迭代的CSP端模型

            for data, target in datasets:# 遍历每个工作机的数据集
                model.send(data.location)# 将模型发送给对应的虚拟机     
                opt.zero_grad()          # 消除之前的梯度
                pred = model(data)       # 预测
                loss = loss_fn(pred, target)     # 计算损失方法为MSE
                loss.backward()            # 回传损失，自动求导得到每个参数的梯度
                opt.step()            # 更新参数
                model.get()            # 获取充电站模型
                acc_pred_sum += pred.get().detach().numpy()[0] #为了画图
            acc = (1-abs(acc_pred_sum-acc_tar_sum)/acc_tar_sum)*100
            iter_acc[iter,cs+1] = acc
            acc_pred_sum = 0
            models += [model]    #合并每个充电站的模型 组成一个列表
        iter_acc[iter,0] = iter  #做个序列号，出图方便
        fl_model = fl_wgt_avg(models, model)    #得到全部充电站的信息，第二层加权联邦学习进行
        models = []   #第二次就得清除上面的模型了，要不然累积
    np.savetxt('fedlearn/iter_acc.csv', iter_acc, fmt="%s", delimiter=',')      
    #输出总的模型        
    return fl_model

def data_evaluation(path):                 #数据评估,测试性能
    res_data = spark.read.csv(path)
    res_tmp = res_data.groupby('_c0').count()
    res_tmp = res_tmp.where('count > 2')  #匹配数据
    res_cnt = np.array(res_tmp.collect())
    res_loss = np.zeros([res_tmp.count(),3])
    loss_cnt = 0
    print('The evaluation was beginning')
    for index in res_cnt[:, 0]: 
        eval_tmp = np.array(res_data.where(res_data._c0 == index).collect()) #选择一个车
        eval_tmp = eval_tmp[:,2:4].astype(float)   #原始数据是字符串格式
        input, tar = eval_tmp[:,0].tolist(), [eval_tmp[0,1]]      #输入和输出
        if max_size <= len(input): 
            input = input[0:max_size]
            data_input = torch.tensor(input, requires_grad=True,dtype=torch.float)   #输入数据
            data_tar = torch.tensor(tar, requires_grad=True,dtype=torch.float)
            if(use_gpu):  #是否使用GPU训练
                FL_model_cpu = FL_model.cpu() #将训练好的模型转移到CPU
            else:FL_model_cpu = FL_model
            #result = nn_extract(input, FL_model_cpu) #传统评价方法
            #print(use_gpu,result)
            preds = FL_model_cpu(data_input)
            loss = torch.sqrt((preds - data_tar) ** 2)
            res_loss[loss_cnt,0] = data_tar.data.detach().numpy()
            res_loss[loss_cnt,1] = preds.detach().numpy()
            res_loss[loss_cnt,2] = loss.data.detach().numpy()#返回损失值
            loss_cnt += 1
    return res_loss  #返回损失值

if __name__ == "__main__":
    spark = SparkSession.builder \
            .master("spark://nuosen:7077") \
            .appName("DataExstract") \
            .getOrCreate()
    EV_data = spark.read.csv('/home/yue_gf/fedlearn/datasets.csv')
    EV_data = EV_data.select("_c0", "_c1", "_c2", "_c3","_c4","_c5")\
            .toDF("EV_ID", "date_time", "input",  "tar", "time", "cs_label")
    cs_cnt = np.array(EV_data.groupby('cs_label').count().collect()) #查看多少个充电站
    P_max = float(EV_data.agg({"input": "max"}).collect()[0]['max(input)'])
    P_min = float(EV_data.agg({"input": "min"}).collect()[0]['min(input)'])   #归一化参数
    max_size = 3   #初始化向量参数
    #km = cluster.KMeans(n_clusters=3, init='k-means++', max_iter=10, n_init=1)
    # 调用该对象的聚类方法
    #km.fit(cs)
    hook = sy.TorchHook(torch) #建立torch和syft的联系
    use_gpu = False    #torch.cuda.is_available()  #是否使用GPU计算
    starttime = datetime.datetime.now()  #测量时间
    FL_model = nn_train(EV_data, len(cs_cnt))    #得到训练模型
    print(FL_model.state_dict())
    total_time = datetime.datetime.now()- starttime
    print('The tarining-time is ', total_time)

    '''rootdir = '/home/yue_gf/fedlearn/ceshi/'     #一个模型四个评价
    pathdir = os.listdir(rootdir) 
    for k in range(0,len(pathdir)):            # Process all files
        path = os.path.join(rootdir,pathdir[k]) 
        res_loss = data_evaluation(path)   #模型误差评估
        np.savetxt('/home/yue_gf/fedlearn/ceshiRMSE/'+pathdir[k], res_loss, fmt="%s", delimiter=",")
      
    print(res_loss)'''

    spark.stop()
    print('The runing is ok!')













