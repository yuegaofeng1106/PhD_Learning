# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2021.11.20
# @Author : Coly
# @version: V3.0 测试版
# @Des    : 第一版大量采用数组，导致程序非常慢，第二个版本采用数据帧，还是不快，第三个版本采用rdd和数据帧
#           并且对程序逻辑进行了再组织（很重要），避免冗余，结果还是明显的，再累也得干。
#           spark有三大引擎，spark core、sparkSQL、sparkStreaming，
#           spark core 的关键抽象是 SparkContext、RDD；
#           SparkSQL 的关键抽象是 SparkSession、DataFrame；
#           sparkStreaming 的关键抽象是 StreamingContext、DStream

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os
import datetime
import pandas as pd
import numpy as np

os.environ['JAVA_HOME'] = '/home/yue_gf/jdk'
os.environ["PYSPARK_PYTHON"] = "/home/yue_gf/anaconda3/bin/python3"

def charge_station(Area_cover): #启发式搜索
    CS_DF = pd.read_csv("cs.csv", delimiter=",",names=['a','b','c'])
    CS_DF = CS_DF.iloc[:,[0,0,1,1,2]]
    CS_DF.columns = ["cs_lon_min", "cs_lon_max", "cs_lat_min", "cs_lat_max",'range']
    CS_DF['cs_lon_min'] = CS_DF['cs_lon_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lon_max'] = CS_DF['cs_lon_max'] + CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_min'] = CS_DF['cs_lat_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_max'] = CS_DF['cs_lat_max'] + CS_DF['cs_lon_min']*Area_cover
    return CS_DF

def charging_behavior():
    window = Window.partitionBy("date", "EV_ID").orderBy("date_time")  #窗口函数
    EV_df = data_init.withColumn("delta_time", data_init['delta_time'] - \
        F.lag("delta_time").over(window))              #准备积分,此处的时间不是充电时间
    EV_df = EV_df.withColumn('mileage',EV_df['delta_time']/3600*EV_df['speed']) #路程积分计算
    EV_df = EV_df.withColumn('mileage', F.sum(EV_df['mileage']).over(window))  # km
    EV_df = EV_df.where('speed == 0').where('state == 0')          #速度为0，且不载客

    # 充电信息提取
    for j in range(len(CSs_data)):     
        lon_min, lon_max = CSs_data.loc[j,'cs_lon_min'], CSs_data.loc[j,'cs_lon_max']
        lat_min, lat_max = CSs_data.loc[j,'cs_lat_min'], CSs_data.loc[j,'cs_lat_max']

        EV_data = EV_df.filter('longitude between '+str(lon_min)+' and ' +str(lon_max))
        EV_data = EV_data.filter('latitude between '+str(lat_min)+' and ' +str(lat_max))   # 是否在充电站
        
        EV_data = EV_data.withColumn("delta_time", EV_data['date_time'] - F.lag("date_time").over(window)) #同一个车的时间差
        EV_data = EV_data.withColumn('delta_time', F.sum(EV_data['delta_time']).over(window)/3600) #得到充电时间，单位小时
        EV_data = EV_data.filter('delta_time between '+str(2)+' and ' +str(3)).\
            filter('mileage between '+str(280)+' and ' +str(320))   #时间和路程的判断
        EV_data = EV_data.withColumn('state',F.lit(j+1)).withColumnRenamed('state', 'cs_label')
        if j == 0: EV_data_tmp = EV_data
        else: EV_data_tmp = EV_data_tmp.union(EV_data)  # 一年的充电量

    return EV_data_tmp  #返回每个充电站的年信息

        
if __name__ == "__main__":

    spark = SparkSession.builder \
            .master("spark://nuosen:7077") \
            .appName("DataExstract") \
            .getOrCreate()

    starttime = datetime.datetime.now()
    textlist = spark.sparkContext.textFile('/home/yue_gf/data/*')   #同时处理所有文件
    rdd = textlist.map(lambda x:x.split(","))  #按逗号分割，形成RDD
    Dataframe = spark.createDataFrame(rdd)  #转成数据帧格式进行分析
    data_init = Dataframe.select("_1", "_2", "_3", "_3", "_4", "_4", "_4","_6", "_8")\
            .toDF("EV_ID", "longitude", "latitude", "mileage", "date", "date_time", "delta_time", "speed", "state")  #重新取名

    data_init = data_init.withColumn('date', F.to_date(data_init.date))  #标记日期，年月日
    data_init = data_init.withColumn('date_time',F.unix_timestamp(F.from_utc_timestamp('date_time','+08:00')))  #转成秒
    data_init = data_init.withColumn('delta_time',F.unix_timestamp(F.from_utc_timestamp('delta_time','+08:00'))).\
        where('longitude > 0').where('latitude > 0') # 预处理，删除异常值
    data_init = data_init.sort('date', 'EV_ID', 'date_time')  #原始数据时间顺序是乱的

    CSs_data = charge_station(0.0005)  # CS数据准备
    EV_data_year = charging_behavior()  #统计年的充电数据
    EV_data_year = EV_data_year.dropDuplicates(["EV_ID", "date_time"])
    EV_date_cnt = np.array(EV_data_year.groupby('date', 'cs_label').count().collect())  #统计每个充电站一年的充电信息

    #保存年的充电数据，保存统计信息
    EV_data_year.coalesce(1).write.csv('/home/yue_gf/2019/')
    np.savetxt('csev.csv', EV_date_cnt, fmt="%s", delimiter=",")

    spark.stop()
    timediff = datetime.datetime.now()- starttime     
    print('The runing time is ', timediff)