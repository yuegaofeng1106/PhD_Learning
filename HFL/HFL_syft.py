# -*- coding: utf-8 -*-
# ！~/anaconda3/bin/python3
# @Time   : 2023.7.3
# @Author : Coly
# @version: V3.0
# @school: X_JT_U
# @Des    : 这个程序主要是为了分层次的联邦学习（HFL）所写，更新了训练集重复使用的问题，将数据的处理放在了本地，
#           因此这里只需要进行联邦学习即可。这是一个全新的版本，逻辑清晰，代码简化许多且可读性强。框架是函数体结构。
#           加快速度，非常明显
#           添加了GPU加速，使用前必须激活pysyft环境，确保有cuda
#           代码需要优化的地方还挺多，这个版本只是能完成基本任务

import findspark
findspark.init()
from pyspark.sql import SparkSession
import os
import numpy as np
import torch
import torch.nn as nn
import copy
import datetime
from tqdm import tqdm

os.environ['JAVA_HOME'] = '/home/linux1/jdk'

def find_number_positions(numbers, k): # 找到列表中等于1的数字，并返回它们在列表中的位置：
    positions = []
    for i, num in enumerate(numbers):
        if num == k:
            positions.append(i)
    return positions

def get_numbers_at_positions(numbers, positions):
    new_list = []
    for pos in positions:
        if pos < len(numbers):
            new_list.append(copy.deepcopy(numbers[pos]))
    return new_list

def FedAvg(models_list):  #均值联邦学习
    glob_model = copy.deepcopy(models_list[0])
    for k in glob_model.keys():
        for i in range(1, len(models_list)): # 从第二个开始加
            glob_model[k] += models_list[i][k]
        glob_model[k] = torch.div(glob_model[k], len(models_list))
    return glob_model  #返回均值后的模型

def Selection_Agg(current_Models, wgt_list, last_Models): # 客户端选择之后进行，聚合
    select_standards = 0.0
    Delta_client = []
    example = copy.deepcopy(current_Models[0])  # 取键值
    last_glob = FedAvg(last_Models)  # 计算两次的全局模型
    current_glob = FedAvg(current_Models)
    for k in example.keys(): # 建立标准
        last_glob[k] = torch.sub(current_glob[k], last_glob[k])  # 求解模型差异
        select_rule = torch.norm(torch.norm(example[k], p=2)) # 将结果张量展平
        select_standards = select_standards + select_rule.item()   # 计算所有元素的二范数
    
    for i in range(len(models_list)):
        norm_ev = 0.0
        for k in example.keys():
            last_Models[i][k] = torch.sub(current_Models[i][k], last_Models[i][k])  # 求解模型差异
            norm = torch.norm(torch.norm(last_Models[i][k], p=2))
            norm_ev = norm_ev + norm.item()
        Delta_client.append(norm_ev)

    # 此处得到了每个模型的变化量和总的选择依据
    indices = [i for i, num in enumerate(Delta_client) if num >= select_standards]  # 使用列表推导式筛选出大于阈值的元素的索引值
    print(Delta_client, select_standards)
    print("before and after: {} and {}".format(len(wgt_list), len(indices)))
    ev_models = get_numbers_at_positions(copy.deepcopy(current_Models), indices)  # 找到对应的模型
    ev_wgt = get_numbers_at_positions(wgt_list, indices)  # 找到对应的权值
    each_CS_model = FL_2_WgtAgg(ev_models, ev_wgt)
    return each_CS_model  #返回均值后的模型

def FL1_selection_agg(models_list, ev_char_label, ev_cs_weight, epoch, last_cs_models):
    #print(len(models_list), len(ev_char_label), len(ev_cs_weight))  # 检查数据一致性
    model_cs1 = copy.deepcopy(models_list[0])  # 有序的字典
    CSs_models = [] # 装最后的结果，也就是每个充电站的聚合模
    for k in range(0, 97):
        if k in ev_char_label:  # 测试的时候不一定充电站全
            position = find_number_positions(ev_char_label, k)  
            new_models_list = get_numbers_at_positions(models_list, position) # 找到充电站对应的模型
            wgt_list = get_numbers_at_positions(ev_cs_weight, position)  # 找到对应的权重
            if epoch == 0:
                agg_k_cs = FedAvg(new_models_list)   # 第一次均值聚合
                CSs_models.append(copy.deepcopy(agg_k_cs))
            else:
                old_models_list = get_numbers_at_positions(last_cs_models, position)
                agg_k_cs = Selection_Agg(new_models_list, wgt_list, old_models_list) # 客户端选择之后进行，聚合
                CSs_models.append(copy.deepcopy(agg_k_cs))
        else:
            CSs_models.append(copy.deepcopy(model_cs1))
    last_cs_models = copy.deepcopy(models_list)      # 保留上一次的值 
    return CSs_models, last_cs_models

def FL_2_WgtAgg(models_list, wgt):  #第二层联邦学习, 加权聚合
    wgt = [x / sum(wgt) for x in wgt] # 权值归一化,和为1
    w_glob = copy.deepcopy(models_list[0])  # 有序的字典
    for k in w_glob.keys():
        w_glob[k] = torch.mul(w_glob[k], wgt[0])  #加权第一个模型，后续就都加在这个上面
    
    for i in range(1, len(models_list)):   # 每个模型乘以每个模型对应的权重
        for k in w_glob.keys(): 
            models_list[i][k] = torch.mul(w_glob[k], wgt[i])
            w_glob[k] = torch.add(w_glob[k], models_list[i][k]) # 累积求和
    return w_glob  #返回加权模型
    
def nn_train(train_data_input, train_data_tar, glob_model):   # 本地学习
    optimizer = torch.optim.Adam(glob_model.parameters(), lr=0.01)     # 创建优化器和损失函数
    loss_fn = nn.MSELoss() # 计算损失方法为MSE
    
    optimizer.zero_grad() # 回传损失，自动求导得到每个参数的梯度
    outputs = glob_model(train_data_input.unsqueeze(1))  #这里训练集合中包含训练值和标签
    loss = loss_fn(outputs.squeeze(1), train_data_tar) # 计算损失方法为MSE
    optimizer.step() # 更新参数
    loss.backward() # 回传损失，自动求导得到每个参数的梯度
    return glob_model.state_dict()  # 获取模型

# 创建 GRU 模型类
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru3 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = input.view(len(input), 1, -1)
        output, _ = self.gru1(input)
        output, _ = self.gru2(output)
        output, _ = self.gru3(output)
        output = self.fc(output[:, -1, :])
        return output

# 读取数据集合，返回数据针
def read_data():
    path_ev = '/home/linux1/test/HFL/datasets/EV_datasets/ev_dataset.csv'
    path_cs = '/home/linux1/test/HFL/datasets/CS_datasets/cs_dataset.csv'
    EV_data = spark.read.csv(path_ev)
    CS_data = spark.read.csv(path_cs)
    EV_data = EV_data.select("_c0", "_c1", "_c2", "_c3", "_c5", "_c6")\
            .toDF("EV_ID", "date_label", "input",  "tar", "cs_label", "wgt")
    CS_data = CS_data.select("_c0", "_c1", "_c2")\
        .toDF("cs_label", "date_label", "input")
    return EV_data, CS_data

# 数据预处理，训练集
def precess_data(localdata, len_train):
    localdata = localdata.toPandas().to_numpy()
    train_data_input = localdata[0:len_train,2]  # 获取训练列表
    train_data_input = np.array(list(map(float, train_data_input)))  # 字符串转换成浮点型
    train_data_tar = localdata[0:len_train,3]  # 获取测试列表
    train_data_tar = np.array(list(map(float, train_data_tar)))  # 字符串转换成浮点型
    cs_label = int(localdata[1,4])   # 取充电站标号
    ev_cs_wgt = float(localdata[1,5])   # 获取每个电动车在每个充电站的权重
    # 标准化数据
    mean = train_data_input.mean()
    std = train_data_input.std()
    train_data_input = (train_data_input - mean) / std
    train_data_tar = (train_data_tar - mean) / std
    #print("数据类型", type(train_data_input[1]),train_data_input[50],cs_label)
    train_data_input = torch.FloatTensor(train_data_input) #转变成张量
    train_data_tar = torch.FloatTensor(train_data_tar)
    return train_data_input,train_data_tar,cs_label,ev_cs_wgt


def precess_cs_train(localdata):
    localdata = localdata.toPandas().to_numpy()
    train_data_input = localdata[:80, 2]  # 获取训练列表
    train_data_input = np.array(list(map(float, train_data_input)))  # 字符串转换成浮点型
    train_data_tar = localdata[:80, 2]  # 获取测试列表
    train_data_tar = np.array(list(map(float, train_data_tar)))  # 字符串转换成浮点型
    
    train_data_input = torch.FloatTensor(train_data_input) #转变成张量
    train_data_tar = torch.FloatTensor(train_data_tar)

    mean = train_data_input.mean()
    std = train_data_input.std()
    train_data_input = (train_data_input - mean) / std
    train_data_tar = (train_data_tar - mean) / std

    train_data_input = torch.FloatTensor(train_data_input) #转变成张量
    train_data_tar = torch.FloatTensor(train_data_tar)
    return train_data_input,train_data_tar


def precess_cs_test(localdata):
    localdata = localdata.toPandas().to_numpy()
    test_data_input = localdata[80:, 2]  # 获取训练列表
    test_data_input = np.array(list(map(float, test_data_input)))  # 字符串转换成浮点型
    test_data_tar = localdata[80:, 2]  # 获取测试列表
    test_data_tar = np.array(list(map(float, test_data_tar)))  # 字符串转换成浮点型
    
    test_data_input = torch.FloatTensor(test_data_input) #转变成张量
    test_data_tar = torch.FloatTensor(test_data_tar)

    mean = test_data_input.mean()
    std = test_data_input.std()
    test_data_input = (test_data_input - mean) / std
    test_data_tar = (test_data_tar - mean) / std

    test_data_input = torch.FloatTensor(test_data_input) #转变成张量
    test_data_tar = torch.FloatTensor(test_data_tar)
    return test_data_input,test_data_tar


def data_evaluation(test_inputs, test_act, glob_model):                 #数据评估,测试性能
    glob_model.eval()
    test_pred = glob_model(test_inputs)
    test_pred = test_pred.squeeze(1)
    criterion = nn.MSELoss()
    test_loss = criterion(test_pred, test_act)   # 计算损失
    test_accuracy = torch.mean(torch.abs(test_pred - test_act)) # 计算精度
    return test_loss.item(), test_accuracy.item()

if __name__ == "__main__":
    spark = SparkSession.builder \
            .master("local[*]") \
            .appName("DataExstract") \
            .config("spark.executor.memory", "4g")\
            .config("spark.executor.cores", "10")\
            .getOrCreate()
    EV_data, CS_data = read_data() # 读取数据

    list_evid = EV_data.select("EV_ID").rdd.flatMap(lambda x: x).collect() # 找到电动汽车列表
    list_csid = list(range(0, 97)) # 充电站列表
    
    wgt = np.loadtxt('/home/linux1/test/HFL/wgt.csv')
    glob_model = GRUModel(input_size=1, hidden_size=30, output_size=1) #模型实例化

    num_epochs = 500 #
    CS_loss_array = np.zeros(shape=(num_epochs, len(list_csid)))
    CS_acc_array = np.zeros(shape=(num_epochs, len(list_csid))) #保存充电站的精度和损失

    start_time = datetime.datetime.now()  #记录时间
    for epoch in range(num_epochs):
        print('The round {} is working! The total rounds are {}'.format(epoch, num_epochs))
        models_list, ev_char_label = [], [] # 电动汽车模型列表，清空所有列表中的模型  准备下一次接收
        ev_cs_weight = []  # EV在充电站的权值
        CS_models_list = []  # 充电站的所有模型， 清空所有列表中的模型  准备下一次接收

        for evid in tqdm(list_evid):
            localdata = EV_data.where(EV_data['EV_ID'] == evid)  # 读取一个数据集合
            train_data_input,train_data_tar,cs_label, ev_cs_wgt = precess_data(localdata, 80)  # 预处理数据
            # 输入数据进行训练，返回model
            local_model = nn_train(train_data_input,train_data_tar, copy.deepcopy(glob_model))
            models_list.append(copy.deepcopy(local_model))    # 获得模型参数,里面装的是模型权值
            ev_char_label.append(cs_label)  #将本地模型和对应的充电站编号保存
            ev_cs_weight.append(ev_cs_wgt)  #将本地模型和对应的充电站权值保存

        #这里会将每个电动汽车分给对应的充电站，通过筛选，得到每个充电站的局部模型，共97个子模型,顺序已经排好
        if epoch == 0: last_cs_models = copy.deepcopy(models_list)
        CSs_models, last_cs_models = FL1_selection_agg(models_list, ev_char_label, ev_cs_weight, epoch, last_cs_models) # 返回的是每个充电站的模型
        #w_glob = FedAvg(models_list)
        for cs_id in list_csid:
            localdata = CS_data.where(CS_data['cs_label'] == str(cs_id))  # 读取一个数据集合
            train_data_input,train_data_tar = precess_cs_train(localdata)  # 预处理数据
            w_glob = copy.deepcopy(CSs_models[cs_id])   # 取列表中对应的模型
            #w_glob = copy.deepcopy(w_glob)   # 取列表中对应的模型
            glob_model.load_state_dict(w_glob) # 拷贝模型到全局网络模型
            CS_model = nn_train(train_data_input,train_data_tar, copy.deepcopy(glob_model))
            CS_models_list.append(copy.deepcopy(CS_model))
            
        w_glob = FL_2_WgtAgg(CS_models_list, wgt)  # CSP进行聚合
        glob_model.load_state_dict(w_glob) # 拷贝模型到全局网络模型
        
        loss_list, accuracy_list= [],[]
        for cs_id in list_csid:   # 全局模型在每个CS的测试集上测试
            localdata = CS_data.where(CS_data['cs_label'] == str(cs_id))  # 读取一个数据集合
            test_inputs, test_act = precess_cs_test(localdata) # 测试集合专用
            test_loss, test_accuracy = data_evaluation(test_inputs, test_act, copy.deepcopy(glob_model))  # 边训练 边测试
            loss_list.append(test_loss)
            accuracy_list.append((1-test_accuracy) * 100)  #每个充电站测试一下，汇总起来
        test_loss = np.mean(loss_list)   # 计算平均损失
        test_accuracy = np.mean(accuracy_list)  # 计算平均准确度
        print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))

        CS_loss_array[epoch, cs_id] = test_loss   # 存储结果
        CS_acc_array[epoch, cs_id] = test_accuracy # 存储结果
    
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    #np.savetxt('CS_loss_array.csv', CS_loss_array, delimiter=',')  #保存结果到本地
    #np.savetxt('CS_acc_array.csv', CS_acc_array, delimiter=',')  #保存结果到本地

    print(f"Program runtime: {elapsed_time}")
    spark.stop()  #关闭spark
    print('The runing is final!')  #程序运行结束



  

    
    













