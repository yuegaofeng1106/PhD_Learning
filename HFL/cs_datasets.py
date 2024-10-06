
# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2023.6.29
# @Author : Coly
# @version: V3.0
# @Des    : 为联邦学习制作数据，用于训练测试.

import findspark
findspark.init()
from pyspark.sql import SparkSession
import os
import pandas as pd

os.environ['JAVA_HOME'] = '/home/linux1/jdk'

df = pd.read_csv('/home/linux1/test/HFL/datasets/EV_datasets/ev_dataset.csv')

df.drop(['EV_id','charging1', 'charging_time', 'weight'], axis=1, inplace=True)
sorted_df = df.sort_values(by=['cs_station', 'date'])
# 合并数据
new_df = sorted_df.groupby(['cs_station', 'date'])['charg'].apply(list).to_frame()
new_df['charg'] = new_df['charg'].apply(lambda x: sum(x))

new_df.to_csv('/home/linux1/test/HFL/datasets/CS_datasets/cs_dataset.csv', encoding='utf-8', header=True)
   
print('The runing is ok!')


