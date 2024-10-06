# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2021.11.20
# @Author : Coly
# @version: V3.0
# @Des    : 这个程序是为了改进数据处理慢的问题 主要是textfile方法的引入，
#           主要实现在驻点停留的车的数量。
# IP = 172.16.248.44

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


def charge_station(Area_cover):   #启发式搜索
    CS_DF = pd.read_csv("cs.csv", delimiter=",",names=['a','b','c'])
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
        EV_data_lon = data_init.filter('longitude between '+str(lon_min)+' and ' +str(lon_max))
        EV_data_stag = EV_data_lon.filter('latitude between '+str(lat_min)+' and ' +str(lat_max))
        EV_data_stag = EV_data_stag.withColumn('state',F.lit(j+1)).withColumnRenamed('state', 'cs_label')
        EV_data_stag = EV_data_stag.dropDuplicates(['date', 'cs_label', 'EV_ID'])  #删除每个车的重复点
        if j == 0: EV_data_tmp = EV_data_stag
        else: EV_data_tmp = EV_data_tmp.union(EV_data_stag)  # 一年的充电量
    return EV_data_tmp

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
            .toDF("EV_ID", "longitude", "latitude", "mileage", "date", "date_time", \
                "delta_time", "speed", "state")  #重新取名

    data_init = data_init.withColumn('delta_time',F.unix_timestamp(F.from_utc_timestamp('delta_time','+08:00'))).\
        dropDuplicates(["EV_ID", "date_time"]).where('longitude > 0').where('latitude > 0') # 删除异常值
    data_init = data_init.withColumn('date', F.to_date(data_init.date))  #标记日期，年月日
    data_init = data_init.dropDuplicates(["EV_ID", "date_time"]) #删除重复行
    data_init = data_init.where('speed == 0').where('state == 0')  #驻点的定义

    CSs_data = charge_station(0.0005)  # CS数据
    EV_data_tmp = stagnation()     #统计驻点信息
    EV_date_cnt = np.array(EV_data_tmp.groupby('date', 'cs_label').count().collect())  #统计
    
    #EV_data_tmp.coalesce(1).write.csv('/home/yue_gf/2019_stag/')
    np.savetxt('stagnation.csv', EV_date_cnt, fmt="%s", delimiter=",")
    spark.stop()
    timediff = datetime.datetime.now()- starttime     
    print('The runing time is ', timediff)

    
	
