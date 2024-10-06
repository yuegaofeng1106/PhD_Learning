# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("D:/pythoncode/dataset/EV_data_third.csv", encoding='latin-1')
names = ['EV_ID', 'date', 'chag_vol', 'act_vol', 'charging_time', 'cs_label']
df.columns = names

# 排序 先按照cs_label 再按照 date
sorted_df = df.sort_values(by=['cs_label', 'date'])

# 时间戳转化为日期格式
sorted_df['datetime'] = pd.to_datetime(sorted_df['date'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
sorted_df.drop(['date', 'EV_ID'], axis=1, inplace=True)

# print(len(sorted_df))
# 删除charging_time 为 0 的行
clear_df = sorted_df.drop(sorted_df[sorted_df['charging_time'] == 0.000000].index)
# print(len(clear_df))
# print(clear_df.head())

# 删除列：act_vol charging_time
clear_df.drop(['act_vol', 'charging_time'], axis=1, inplace=True)


print(clear_df.dtypes)
# 新增列 day hour
clear_df['day'] = clear_df['datetime'].apply(lambda x: x[0:10])
clear_df['hour'] = clear_df['datetime'].apply(lambda x: x[11:13])

# 删除列：datetime
clear_df.drop(['datetime'], axis=1, inplace=True)
print(clear_df.head())

group_df = clear_df.groupby(['cs_label', 'day', 'hour'])
result = group_df.sum()
# new_df = sorted_df.groupby(['cs_station', 'date'])['charg'].apply(list).to_frame()
# new_df['charg'] = new_df['charg'].apply(lambda x: sum(x))
print(result.head())
result.to_csv('./cs_third.csv',  encoding='utf-8', header=True)