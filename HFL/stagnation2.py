# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2021.11.1
# @Author : Coly
# @version: V2.0
# @Des    : None

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
import numpy as np
import pandas as pd
import datetime


os.environ['JAVA_HOME'] = '/home/yue_gf/jdk'
os.environ["PYSPARK_PYTHON"] = "/home/yue_gf/anaconda3/bin/python3"

def charge_station(Area_cover):
    CS_DF = pd.read_csv("/home/yue_gf/cs.csv", delimiter=",",names=['a','b','c'])
    CS_DF = CS_DF.iloc[:,[0,0,1,1,2]]
    CS_DF.columns = ["cs_lon_min", "cs_lon_max", "cs_lat_min", "cs_lat_max",'range']
    CS_DF['cs_lon_min'] = CS_DF['cs_lon_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lon_max'] = CS_DF['cs_lon_max'] + CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_min'] = CS_DF['cs_lat_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_max'] = CS_DF['cs_lat_max'] + CS_DF['cs_lon_min']*Area_cover
    return CS_DF

def stagnation():
    for j in range(len(CSs_data)):
        lon_min, lon_max = CSs_data.loc[j,'cs_lon_min'], CSs_data.loc[j,'cs_lon_max']
        lat_min, lat_max = CSs_data.loc[j,'cs_lat_min'], CSs_data.loc[j,'cs_lat_max']
        EV_data_lon = EV_data.filter('longitude between '+str(lon_min)+' and ' +str(lon_max))
        EV_data_stag = EV_data_lon.filter('latitude between '+str(lat_min)+' and ' +str(lat_max))
        EV_count = len(EV_data_stag.groupby('EV_ID').count().collect())
        if data_col == 1: cs_stagnation[j,0], cs_stagnation[j,data_col] = j, int(EV_count)
        else: cs_stagnation[j,data_col] = int(EV_count)  #每次编号浪费计算，编号进行一次，能快一点就快一点

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    spark = SparkSession.builder \
            .master("spark://nuosen:7077") \
            .appName("DataExstract") \
             .getOrCreate()

    starttime = datetime.datetime.now()
    textlist = spark.sparkContext.textFile('/home/yue_gf/data/*')
    rdd = textlist.map(lambda x:x.split(","))  #按逗号分割，形成RDD
    Dataframe = spark.createDataFrame(rdd)  #转成数据帧格式进行分析
    data_init = Dataframe.select("_1", "_2", "_3", "_3", "_4","_4","_6", "_8")\
            .toDF("EV_ID", "longitude", "latitude", "mileage", "date_time", "delta_time", "speed", "state")  #重新取名

    data_init = data_init.withColumn('delta_time',F.unix_timestamp(F.from_utc_timestamp('delta_time','+08:00'))).\
        dropDuplicates(["EV_ID", "date_time"]).where('longitude > 0').where('latitude > 0') # 删除异常值
    data_init = data_init.withColumn('date_time', F.to_date(data_init.date_time))  #标记日期，年月日
    data_init = data_init.where('speed == 0').where('state == 0')  #驻点的定义

    CSs_data = charge_station(0.0005)  # CS数据准备
    EV_date_cnt = np.array(data_init.groupby('date_time').count().collect())  #统计都是哪些日期
    cs_stagnation = np.ones([len(CSs_data),len(EV_date_cnt[:, 0])+1]).astype(int) # 保存统计结果
    data_col = 1  #一天一列
    for index in EV_date_cnt[:, 0]:
        EV_data = data_init.where(data_init.date_time == index)  #选择一天的时间
        stagnation()  #统计每天的驻点数据并且保存
        data_col += 1 #保存数据的时候往后移动一列 

    np.savetxt('stagnation.csv', cs_stagnation, fmt="%d", delimiter=",")

    spark.stop()
    timediff = datetime.datetime.now()- starttime     
    print('The runing time is ', timediff)

    
	
