#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
from .utils import Distance

'''抽取wifi相关的特征'''
DIVIDE_PROTECT = 0.01

# wifi强度的均值,最大值,最小值
def get_wifi_feature1(result):
    row_list = []
    for _,row in result.iterrows():
        wifi_list = [wifi.split('|') for wifi in str(row['wifi_infos']).split(';')]
        max_score = -113 # 最小值
        min_score = -1 # 最大值
        average_score = 0 # 平均值
        count = 0
        link = 0  # 0代表没有连接,1代表连接
        link_wifi_name = 'none' # 随便取的
        link_wifi_score = -113
        for i in wifi_list:
            try:
                if str(i[2]) == 'true':
                    link = 1
                    link_wifi_name = str(i[0])
                    link_wifi_score = int(i[1])
                if int(i[1]) > max_score:
                    max_score = int(i[1])
                if int(i[1]) < min_score:
                    min_score = int(i[1])
                average_score += int(i[1])
                count += 1
            except IndexError:
                print(i)
        average_score /= (count+DIVIDE_PROTECT)
        row['feature_wifi_max_score'] = max_score
        row['feature_wifi_min_score'] = min_score
        row['feature_wifi_average_score'] = average_score
        # 以下不是特征,便于后期计算
        row['wifi_if_linked'] = link
        row['wifi_numbers'] = count
        row['link_wifi_name'] = link_wifi_name
        row['link_wifi_score'] = link_wifi_score
        row_list.append(row)
    result = pd.DataFrame(row_list)
    return result


def get_wifi_feature2(result,config,train = True):
    row_list = []
    if train:
        shop_dataframe = pd.read_csv(config.pre_shop_wifi)
    else:
        shop_dataframe = pd.read_csv(config.after_shop_wifi)
    result = pd.merge(result,shop_dataframe,on='shop_id',how='left')

    columns_raw = set(result.columns)
    refer_shop_set = set(shop_dataframe.shop_id)
    for index,row in result.iterrows():
        if row['shop_id'] in refer_shop_set:
            user_wifi_dict = {} # key是bssid,value是强度
            for wifi in str(row['wifi_infos']).split(';'):
                wifi_list = wifi.split('|')
                try:
                    user_wifi_dict[wifi_list[0].strip()] = int(wifi_list[1])
                except IndexError:
                    pass
            user_wifi_dict = sorted(user_wifi_dict.items(), key=lambda v:v[1],reverse = True)
            user_wifi_name = [key[0] for key in user_wifi_dict] # 得到有序的wifi名
            user_wifi_strength = [key[1] for key in user_wifi_dict] # 得到有序的wifi强度
            user_wifi_rank = [i+1 for i in range(len(user_wifi_name))]

            user_wifi_length = len(user_wifi_name)
            shop_wifi_name = [wifi for wifi in str(row['wifi']).split() if wifi != 'nan'][:user_wifi_length]
            shop_wifi_rank_mode =[float(rank) for rank in str(row['wifi_rank_mode']).split() if rank != 'nan'][:user_wifi_length] # 得到rank众数
            shop_wifi_rank_average = [float(rank) for rank in str(row['wifi_rank_average']).split() if rank != 'nan'][:user_wifi_length] # 平均数
            # shop_wifi_rank_max = [float(rank) for rank in str(row['wifi_rank_max']).split() if rank != 'nan'][:user_wifi_length] # rank最大值
            # shop_wifi_rank_min = [float(rank) for rank in str(row['wifi_rank_min']).split() if rank != 'nan'][:user_wifi_length] # rank最小值
            shop_wifi_strength_mode = [float(rank) for rank in str(row['wifi_strength_mode']).split() if rank != 'nan'][:user_wifi_length]
            shop_wifi_strength_average = [float(rank) for rank in str(row['wifi_strength_average']).split() if rank != 'nan'][:user_wifi_length]
            # shop_wifi_strength_max = [float(rank) for rank in str(row['wifi_strength_max']).split() if rank != 'nan'][:user_wifi_length]
            # shop_wifi_strength_min = [float(rank) for rank in str(row['wifi_strength_min']).split() if rank != 'nan'][:user_wifi_length]
            shop_wifi_count = [int(wifi_count) for wifi_count in str(row['wifi_count']).split() if wifi_count != 'nan'][:user_wifi_length]
            main_wifi_all_count = sum(shop_wifi_count)

            # 特征1,交集，逆序数
            common_list_features = Distance.common_set_number_and_ratio(user_wifi_name,shop_wifi_name)
            row['feature_common_wifi_count'] = common_list_features[0] # 交集数量
            row['feature_common_wifi_count_shop_ratio'] = common_list_features[1] # 交集比例
            row['feature_common_wifi_count_user_shop_all_ratio'] = common_list_features[2] # 交集/并集
            row['feature_common_wifi_inverse_pairs'] = common_list_features[3] # 交集逆序数
            row['feature_common_wifi_inverse_pairs_ratio'] = common_list_features[4] # 交集逆序数/交集

            # 特征2,lcs
            row['feature_wifi_lcs'] = Distance.lcs_length(user_wifi_name,shop_wifi_name) # lcs长度

            # 特征3,曼哈顿,信号强度,众数
            manhattan_features1 = Distance.manhattan_distance(user_wifi_name,user_wifi_strength,
                                                              shop_wifi_name,shop_wifi_strength_mode)
            row['feature_wifi_strength_mode_manhatten_in'] =  manhattan_features1[0]
            row['feature_wifi_strength_mode_manhatten_out'] =  manhattan_features1[1]
            row['feature_wifi_strength_mode_manhatten_all'] = manhattan_features1[2]
            row['feature_wifi_strength_mode_manhatten_all_punish'] = manhattan_features1[3]
            row['feature_wifi_strength_mode_manhatten_in_out_ratio'] = manhattan_features1[4]
            row['feature_wifi_strength_mode_manhatten_in_all_ratio'] = manhattan_features1[5]
            row['feature_wifi_strength_mode_manhatten_in_all_punish_ratio'] = manhattan_features1[6]

            # 特征4,曼哈顿,信号强度,平均数
            manhattan_features2 = Distance.manhattan_distance(user_wifi_name,user_wifi_strength,
                                                              shop_wifi_name,shop_wifi_strength_average)
            row['feature_wifi_strength_avr_manhatten_in'] =  manhattan_features2[0]
            row['feature_wifi_strength_avr_manhatten_out'] =  manhattan_features2[1]
            row['feature_wifi_strength_avr_manhatten_all'] = manhattan_features2[2]
            row['feature_wifi_strength_avr_manhatten_all_punish'] = manhattan_features2[3]
            row['feature_wifi_strength_avr_manhatten_in_out_ratio'] = manhattan_features2[4]
            row['feature_wifi_strength_avr_manhatten_in_all_ratio'] = manhattan_features2[5]
            row['feature_wifi_strength_avr_manhatten_in_all_punish_ratio'] = manhattan_features2[6]

            # 特征5，曼哈顿，rank，众数
            manhattan_features3 = Distance.manhattan_distance(user_wifi_name,user_wifi_rank,
                                                              shop_wifi_name,shop_wifi_rank_mode)
            row['feature_wifi_rank_mode_manhatten_in'] =  manhattan_features3[0]
            row['feature_wifi_rank_mode_manhatten_out'] =  manhattan_features3[1]
            row['feature_wifi_rank_mode_manhatten_all'] = manhattan_features3[2]
            row['feature_wifi_rank_mode_manhatten_all_punish'] = manhattan_features3[3]
            row['feature_wifi_rank_mode_manhatten_in_out_ratio'] = manhattan_features3[4]
            row['feature_wifi_rank_mode_manhatten_in_all_ratio'] = manhattan_features3[5]
            row['feature_wifi_rank_mode_manhatten_in_all_punish_ratio'] = manhattan_features3[6]

            # 特征6,曼哈顿,rank,平均数
            manhattan_features4 = Distance.manhattan_distance(user_wifi_name,user_wifi_rank,
                                                              shop_wifi_name,shop_wifi_rank_average)
            row['feature_wifi_rank_avr_manhatten_in'] =  manhattan_features4[0]
            row['feature_wifi_rank_avr_manhatten_out'] =  manhattan_features4[1]
            row['feature_wifi_rank_avr_manhatten_all'] = manhattan_features4[2]
            row['feature_wifi_rank_avr_manhatten_all_punish'] = manhattan_features4[3]
            row['feature_wifi_rank_avr_manhatten_in_out_ratio'] = manhattan_features4[4]
            row['feature_wifi_rank_avr_manhatten_in_all_ratio'] = manhattan_features4[5]
            row['feature_wifi_rank_avr_manhatten_in_all_punish_ratio'] = manhattan_features4[6]

            # 特征7,欧几里得,信号强度,众数
            euclidean_features1 = Distance.euclidean_distance(user_wifi_name,user_wifi_strength,
                                                              shop_wifi_name,shop_wifi_strength_mode)
            row['feature_wifi_strength_mode_euclidean_in'] =  euclidean_features1[0]
            row['feature_wifi_strength_mode_euclidean_out'] =  euclidean_features1[1]
            row['feature_wifi_strength_mode_euclidean_all'] = euclidean_features1[2]
            row['feature_wifi_strength_mode_euclidean_all_punish'] = euclidean_features1[3]
            row['feature_wifi_strength_mode_euclidean_in_out_ratio'] = euclidean_features1[4]
            row['feature_wifi_strength_mode_euclidean_in_all_ratio'] = euclidean_features1[5]
            row['feature_wifi_strength_mode_euclidean_in_all_punish_ratio'] = euclidean_features1[6]

            # 特征8,欧几里得,信号强度,平均数
            euclidean_features2 = Distance.euclidean_distance(user_wifi_name,user_wifi_strength,
                                                              shop_wifi_name,shop_wifi_strength_average)
            row['feature_wifi_strength_avr_euclidean_in'] =  euclidean_features2[0]
            row['feature_wifi_strength_avr_euclidean_out'] =  euclidean_features2[1]
            row['feature_wifi_strength_avr_euclidean_all'] = euclidean_features2[2]
            row['feature_wifi_strength_avr_euclidean_all_punish'] = euclidean_features2[3]
            row['feature_wifi_strength_avr_euclidean_in_out_ratio'] = euclidean_features2[4]
            row['feature_wifi_strength_avr_euclidean_in_all_ratio'] = euclidean_features2[5]
            row['feature_wifi_strength_avr_euclidean_in_all_punish_ratio'] = euclidean_features2[6]

            # 特征9，欧几里得，rank，众数
            euclidean_features3 = Distance.euclidean_distance(user_wifi_name,user_wifi_rank,
                                                              shop_wifi_name,shop_wifi_rank_mode)
            row['feature_wifi_rank_mode_euclidean_in'] =  euclidean_features3[0]
            row['feature_wifi_rank_mode_euclidean_out'] =  euclidean_features3[1]
            row['feature_wifi_rank_mode_euclidean_all'] = euclidean_features3[2]
            row['feature_wifi_rank_mode_euclidean_all_punish'] = euclidean_features3[3]
            row['feature_wifi_rank_mode_euclidean_in_out_ratio'] = euclidean_features3[4]
            row['feature_wifi_rank_mode_euclidean_in_all_ratio'] = euclidean_features3[5]
            row['feature_wifi_rank_mode_euclidean_in_all_punish_ratio'] = euclidean_features3[6]

            # 特征10,欧几里得,rank,平均数
            euclidean_features4 = Distance.euclidean_distance(user_wifi_name,user_wifi_rank,
                                                              shop_wifi_name,shop_wifi_rank_average)
            row['feature_wifi_rank_avr_euclidean_in'] =  euclidean_features4[0]
            row['feature_wifi_rank_avr_euclidean_out'] =  euclidean_features4[1]
            row['feature_wifi_rank_avr_euclidean_all'] = euclidean_features4[2]
            row['feature_wifi_rank_avr_euclidean_all_punish'] = euclidean_features4[3]
            row['feature_wifi_rank_avr_euclidean_in_out_ratio'] = euclidean_features4[4]
            row['feature_wifi_rank_avr_euclidean_in_all_ratio'] = euclidean_features4[5]
            row['feature_wifi_rank_avr_euclidean_in_all_punish_ratio'] = euclidean_features4[6]

            # 特征11,余弦相似度
            row['feature_wifi_rank_avr_cos_similarity'] = Distance.cosine_similarity(user_wifi_rank,shop_wifi_rank_average)
            row['feature_wifi_rank_mode_cos_similarity'] = Distance.cosine_similarity(user_wifi_rank,shop_wifi_rank_mode)
            row['feature_wifi_strength_avr_cos_similarity'] = Distance.cosine_similarity(user_wifi_strength,
                                                                                         shop_wifi_strength_average)
            row['feature_wifi_strength_mode_cos_similarity'] = Distance.cosine_similarity(user_wifi_strength,
                                                                                          shop_wifi_strength_mode)

            # 特征12，match score
            row['feature_wifi_match_score'] = Distance.match_score(user_wifi_name,shop_wifi_name,shop_wifi_count)

            # 特征13，计算wifi连接的特征
            if int(row['wifi_if_linked']) == 1: # 当用户连接到wifi时,判断是否
                shop_wifi_dict = {}
                for i in range(len(shop_wifi_name)):
                    shop_wifi_dict[shop_wifi_name[i]] = shop_wifi_count[i]
                if row['link_wifi_name'] in shop_wifi_name:
                    row['feature_wifi_link_shop'] = 1
                    row['feature_wifi_link_shop_score_ratio'] = shop_wifi_dict[row['link_wifi_name']]/ main_wifi_all_count
                else:
                    row['feature_wifi_link_shop'] = 0
                    row['feature_wifi_link_shop_score_ratio'] = 0.0
            else:
                row['feature_wifi_link_shop'] = 0
                row['feature_wifi_link_shop_score_ratio'] = 0.0
        # 店铺不在参照集店铺表中
        else:
            pass
        row_list.append(row)

    result = pd.DataFrame(row_list)

    # 填充缺失值
    columns_all = set(result.columns)
    columns_diff = columns_all - columns_raw # 新增加的列
    for column in columns_diff:
        result[column].fillna(result[column].mean(),inplace=True)

    del result['wifi_if_linked']
    del result['wifi_numbers']
    del result['link_wifi_name']
    del result['link_wifi_score']

    return result
