# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2021.11.1
# @Author : Coly
# @version: V1.0
# @Des    : draw metri 1

import findspark
findspark.init()
from pyspark.sql import SparkSession, functions
import os
import numpy as np
import pandas as pd
import datetime

os.environ['JAVA_HOME'] = '/home/yue_gf/jdk'
os.environ["PYSPARK_PYTHON"] = "/home/yue_gf/anaconda3/bin/python3"
# IP = 172.16.248.44

def charge_station(Area_cover):
    CS_DF = pd.read_csv("/home/yue_gf/cs.csv", delimiter=",",names=['a','b','c'])
    CS_DF = CS_DF.iloc[:,[0,0,1,1,2]]
    CS_DF.columns = ["cs_lon_min", "cs_lon_max", "cs_lat_min", "cs_lat_max",'range']
    CS_DF['cs_lon_min'] = CS_DF['cs_lon_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lon_max'] = CS_DF['cs_lon_max'] + CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_min'] = CS_DF['cs_lat_min'] - CS_DF['cs_lon_min']*Area_cover
    CS_DF['cs_lat_max'] = CS_DF['cs_lat_max'] + CS_DF['cs_lon_min']*Area_cover
    return CS_DF

def stagnation(path):
    EV_DF = spark.read.csv(path)    #local reading file
    EV_data = EV_DF.select("_c0", "_c1", "_c2", "_c2","_c3","_c3", "_c5", "_c7")\
            .toDF("EV_ID", "longitude", "latitude", "mileage", "date_time", "delta_time", "speed", "state")  # renamed for column
    EV_data = EV_data.where('speed == 0').where('state == 0') # The speed of charging EV is zero and state is zero
    # Get stagnation
    for j in range(len(CSs_data)):
        lon_min, lon_max = CSs_data.loc[j,'cs_lon_min'], CSs_data.loc[j,'cs_lon_max']
        lat_min, lat_max = CSs_data.loc[j,'cs_lat_min'], CSs_data.loc[j,'cs_lat_max']
        EV_data_lon = EV_data.filter('longitude between '+str(lon_min)+' and ' +str(lon_max))
        EV_data_stag = EV_data_lon.filter('latitude between '+str(lat_min)+' and ' +str(lat_max))
        EV_count = len(EV_data_stag.groupby('EV_ID').count().collect())
        cs_stagnation[j,0],cs_stagnation[j,k+1] = j+1, int(EV_count) 

#172.16.248.44  .master("spark://172.16.248.44:7077")\
# .master("local[*]")
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    spark = SparkSession.builder \
            .master("spark://nuosen:7077") \
            .appName("DataExstract") \
            .getOrCreate()
    CSs_data = charge_station(0.0005)  # parameter is size of cs
    rootdir = '/home/yue_gf/data/'
    pathdir = sorted(os.listdir(rootdir))
    print(pathdir)
    cs_stagnation = np.ones([len(CSs_data),len(pathdir)+1]).astype(int) # Storage results
     
    for k in range(0,len(pathdir)):            # Process all files  
        path = os.path.join(rootdir,pathdir[k])
        print(k+1, path)
        stagnation(path)

    np.savetxt('stagnation.csv', cs_stagnation, fmt="%d", delimiter=",")
    
    spark.stop()

    timediff = datetime.datetime.now()- starttime     
    print('The runing time is ', timediff)

    
	
