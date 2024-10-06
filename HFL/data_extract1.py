# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2021.10.28
# @Author : Coly
# @version: V1.0
# @Des    : data analysis.  spark有三大引擎，spark core、sparkSQL、sparkStreaming，
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

def charge_station(Area_cover):
    CS_DF = pd.read_csv("/home/yue_gf/cs.csv", delimiter=",",names=['a','b','c'])
    CS_DF = CS_DF.iloc[:,[0,0,1,1,2]]
    CS_DF.columns = ["cs_lon_min", "cs_lon_max", "cs_lat_min", "cs_lat_max",'range']
    CS_DF['cs_lon_min'] = CS_DF['cs_lon_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lon_max'] = CS_DF['cs_lon_max'] + CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_min'] = CS_DF['cs_lat_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_max'] = CS_DF['cs_lat_max'] + CS_DF['cs_lon_min']*Area_cover
    return CS_DF

def charging_behavior(path):
    EV_DF = spark.read.csv(path)    #local reading file
    
    EV_df = EV_DF.select("_c0", "_c1", "_c2", "_c2","_c3","_c3", "_c5", "_c7")\
            .toDF("EV_ID", "longitude", "latitude", "mileage", "date_time", "delta_time", "speed", "state")  # renamed for column
    EV_df = EV_df.where('longitude > 0').where('latitude > 0') # The data exist error
    
    EV_df = EV_df.withColumn('date_time',F.unix_timestamp(F.from_utc_timestamp('date_time','+08:00')))#Converted to seconds, convenient for difference and integration 
    EV_df = EV_df.withColumn('delta_time',F.unix_timestamp(F.from_utc_timestamp('delta_time','+08:00')))
      
    EV_df = EV_df.sort('EV_ID', 'date_time')  #Joint sorting method:Get the time sequence of the same car

    window = Window.partitionBy("EV_ID").orderBy("date_time")
    EV_df = EV_df.withColumn("delta_time", EV_df['delta_time'] - F.lag("delta_time").over(window)).where("delta_time > 0")  #Take a row of values down
    
    EV_df = EV_df.withColumn('mileage',EV_df['delta_time']/3600*EV_df['speed']) #Calculate velocity integral
    EV_df = EV_df.withColumn('mileage', F.sum(EV_df['mileage']).over(window))  # km
    EV_df = EV_df.where('speed == 0').where('state == 0') # The speed of charging EV is zero and state is zero

    Schema = ["EV_ID", "longitude", "latitude", "mileage", "date_time", "delta_time", "speed", "state"]
    values = [(0,0,0,0,0,0,0,0)]
    EV_data_tmp = spark.createDataFrame(values,Schema) # An empty dataframe will cause an error
    # Get real charging information
    for j in range(len(CSs_data)):
        
        lon_min, lon_max = CSs_data.loc[j,'cs_lon_min'], CSs_data.loc[j,'cs_lon_max']
        lat_min, lat_max = CSs_data.loc[j,'cs_lat_min'], CSs_data.loc[j,'cs_lat_max']

        EV_data = EV_df.filter('longitude between '+str(lon_min)+' and ' +str(lon_max))
        EV_data = EV_data.filter('latitude between '+str(lat_min)+' and ' +str(lat_max))   # Is it in the charging station

        EV_data = EV_data.withColumn("delta_time",EV_data['date_time'] - F.lag("date_time").over(window)) #Take a row of values down
        EV_data = EV_data.withColumn('delta_time', F.sum(EV_data['delta_time']).over(window)/3600) #The time difference becomes an hour
        EV_data = EV_data.filter('delta_time between '+str(2)+' and ' +str(3)).\
            filter('mileage between '+str(200)+' and ' +str(250)).\
                dropDuplicates(['EV_ID'])    #Judge driving mileage
        EV_data = EV_data.withColumn('state',F.lit(j)).withColumnRenamed('state', 'cs_label')
        csev_number[j,0],csev_number[j,k+1] = j+1, int(EV_data.count()) # Number of selected vehicles
        EV_data_tmp = EV_data_tmp.union(EV_data)  # charging of one day
    #preserve data using coalesce(joint),save charging time
    #EV_data_tmp.coalesce(1).write.csv(''/home/yue_gf/Charge_EV/Data_2019030' + pathdir[k])
        

if __name__ == "__main__":

    starttime = datetime.datetime.now()

    spark = SparkSession.builder \
            .master("spark://nuosen:7077") \
            .appName("DataExstract") \
            .getOrCreate()

    CSs_data = charge_station(0.0005)  #parameter is size of cs

    rootdir = '/home/yue_gf/data/'
    pathdir = sorted(os.listdir(rootdir))

    csev_number = np.ones([len(CSs_data),len(pathdir)+1]).astype(int) # Storage results

    for k in range(0,len(pathdir)):            # Process all files  
        path = os.path.join(rootdir,pathdir[k])
        print(k+1, path)
        charging_behavior(path)

    np.savetxt('csev_number.csv', csev_number, fmt="%d", delimiter=',')
    spark.stop()
    timediff = datetime.datetime.now()- starttime     
    print('The runing time is ', timediff)


    
	
