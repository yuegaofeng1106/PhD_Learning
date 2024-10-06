# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("./cs_third.csv")
print(df.head())

grouped_df = df.groupby(['cs_label', 'day'])  # 按 cs_label 和 day进行分组

group_map = grouped_df.groups
#print(group_map)
hours = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
# temp_df = grouped_df.get_group((0, '2019/1/1'))
# print(temp_df)
# need_hour = list(hours - set(temp_df['hour'].tolist()))
# print(need_hour)
# #
# new_row = {'cs_label':0, 'day':'2019/1/1', 'hour':0, 'chag_vol':0.0}
# temp_df = temp_df.append(new_row, ignore_index=True)
# # temp_df.loc[0] = [0, '2019/1/1', 0, 0]
res_df = pd.DataFrame(columns=['cs_label', 'day', 'hour', 'chag_vol'])

for key in group_map.keys():
    temp_df = grouped_df.get_group(key)
    need_hour = list(hours - set(temp_df['hour'].tolist()))
    for need in need_hour:
        new_row = {'cs_label': key[0], 'day': key[1], 'hour': need, 'chag_vol': 0}
        temp_df = temp_df.append(new_row, ignore_index=True)
    sorted_df = temp_df.sort_values(by=['hour'])
    res_df = pd.concat([res_df, sorted_df], ignore_index=True)
res_df.to_csv('./cs_third_new.csv', encoding='utf-8', header=True)
#res_df.to_csv('./cs_station_second.csv', encoding='utf-8', header=True)