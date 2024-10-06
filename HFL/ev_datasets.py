
# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2023.6.29
# @Author : Coly
# @version: V3.0
# @Des    : 为联邦学习制作数据，归一化放在这里，用于训练测试.

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os
from collections import Counter
import numpy as np
import random

os.environ['JAVA_HOME'] = '/home/linux1/jdk'

def find_most_common_element(data_list):
    # 使用Counter类统计元素出现的次数
    count_dict = Counter(data_list)
    total = sum(count_dict.values())
    # 找到最大的概率值
    probabilities = {key: value / total for key, value in count_dict.items()}
    max_probability = max(probabilities.values())
    max_probability = round(max_probability, 4)
    # 找到出现次数最多的元素
    most_common_element = count_dict.most_common(1)[0][0]
    return most_common_element, max_probability

if __name__ == "__main__":
    spark = SparkSession.builder \
            .master("local[*]") \
            .appName("DataExstract") \
            .config("spark.executor.memory", "4g")\
            .config("spark.executor.cores", "10")\
            .getOrCreate()  
    Schema = ["EV_ID", "date", "chag_vol",  "act_vol", "charging_time", "cs_label", "weight"]
    values = [(0,0,0,0,0,0,0)]
    EV_year = spark.createDataFrame(values,Schema) # An empty dataframe will cause an error
    rootdir = '/home/linux1/test/Charge_EV/'
    pathdir = sorted(os.listdir(rootdir))
    for k in range(0,len(pathdir)):            # Process files  #len(pathdir)
        path = os.path.join(rootdir,pathdir[k])
        print(k+1,path)
        EV_day = spark.read.csv(path)
        EV_day = EV_day.select("_c0", "_c4", "_c3", "_c3", "_c5", "_c7", "_c7")
        EV_year = EV_year.union(EV_day)  #get data
    EV_year = EV_year.where('date > 0')
    EV_year = EV_year.withColumn("chag_vol", EV_year['chag_vol'] * 0.067)  #电能和里程转换，单位：度
    window = Window.partitionBy("EV_ID").orderBy('EV_ID')
    EV_year = EV_year.withColumn("act_vol", F.avg("chag_vol").over(window))
    EV_year = EV_year.withColumn("date",F.from_unixtime('date', format="yyyy-MM-dd"))
    EV_year = EV_year.dropDuplicates(['EV_ID', "date", "chag_vol",  "act_vol", "charging_time"])

    # 读取EV的标签
    ev_id = spark.read.csv('/home/linux1/test/HFL/datasets/EV_datasets/ev_dataset_id.csv')
    list_evid = ev_id.select("_c0").rdd.flatMap(lambda x: x).collect()

    Schema = ["EV_ID", "date", "charg_vol",  "act_vol", "charging_time", "cs_label", "weight"]
    values = [(0,0,0,0,0,0,0)]
    EV_final = spark.createDataFrame(values,Schema) # An empty dataframe will cause an error
    day_number = 90
    for i in range(0, EV_year.count(), day_number):
        temp_df = EV_year.limit(day_number).toPandas()
        EV_year = EV_year.drop(str(range(day_number)))  # 立马删除处理完的
        temp_df = spark.createDataFrame(temp_df, EV_year.schema) #需要处理的
        cs_label_list = random_list = [random.randrange(0, 97) for _ in range(97)] #充电站
        Popular, Probability = find_most_common_element(cs_label_list)  # 找到概率最大值
        #print(Popular, Probability)
        temp_df = temp_df.withColumn("weight", F.lit(str(Probability)))  # 操作权重
        temp_df = temp_df.withColumn("cs_label", F.lit(str(Popular)))  # 操作充电站
        temp_df = temp_df.withColumn("EV_ID", F.lit(list_evid[i]))  #EV ID
        EV_final = EV_final.union(temp_df)

    EV_final = EV_final.filter(EV_final.EV_ID != '0')
    window=Window.partitionBy('EV_ID').orderBy('EV_ID')
    EV_final = EV_final.withColumn('date', F.row_number().over(window))  # 设置日期
    EV_final = EV_final.withColumn("cha_vol", EV_final["act_vol"])
    EV_final = EV_final.select("EV_ID", "date", "cha_vol",  "act_vol", "charging_time", "cs_label", "weight")
    #EV_final.show(5)
    EV_final.coalesce(1).write.csv('/home/linux1/test/HFL/datasets//EV_datasets/ev_dataset.csv')

    spark.stop()
    print('The runing is ok!')


