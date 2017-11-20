#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from utils import euclidean_distance,manhattan_distance,haversine_distance

'''获取地理位置相关特征'''

# 用户和店铺的距离
def get_user_shop_distance(result):
    result['feature_user_shop_lon_sub'] = (result['user_longitude'] - result['shop_longitude'])
    result['feature_user_shop_lat_sub'] = (result['user_latitude'] - result['shop_latitude'])
    result['feature_user_shop_lon_sub_abs'] = abs(result['feature_user_shop_lon_sub'])
    result['feature_user_shop_lat_sub_abs'] = abs(result['feature_user_shop_lat_sub'])
    result['feature_user_shop_uclidean_dis'] = euclidean_distance(result['user_latitude'],result['user_longitude'],result['shop_latitude'],result['shop_longitude'])
    result['feature_user_shop_haversine_dis'] = haversine_distance(result['user_latitude'],result['user_longitude'],result['shop_latitude'],result['shop_longitude'])
    result['feature_user_shop_manhattan_dis'] = manhattan_distance(result['user_latitude'],result['user_longitude'],result['shop_latitude'],result['shop_longitude'])
    return result


# 用户距离和在该店铺消费的位置的距离
def get_user_shop_average_distance(refer,result):
    shop_longitude = refer.groupby(['shop_id'],as_index = False)['longitude'].agg({'shop_average_longitude':'mean'})
    shop_latitude = refer.groupby(['shop_id'],as_index = False)['latitude'].agg({'shop_average_latitude':'mean'})
    result = pd.merge(result,shop_longitude,on=['shop_id'],how='left')
    result = pd.merge(result,shop_latitude,on=['shop_id'],how='left')
    result['feature_user_shop_aver_uclidean_dis'] = euclidean_distance(result['user_latitude'],result['user_longitude'],result['shop_average_latitude'],result['shop_average_longitude'])
    result['feature_user_shop_aver_haversine_dis'] = haversine_distance(result['user_latitude'],result['user_longitude'],result['shop_average_latitude'],result['shop_average_longitude'])
    result['feature_user_shop_aver_manhattan_dis'] = manhattan_distance(result['user_latitude'],result['user_longitude'],result['shop_average_latitude'],result['shop_average_longitude'])
    del result['shop_average_longitude']
    del result['shop_average_latitude']
    return result

# 用户和店铺的斜率
def get_user_shop_slope(result):
    result['feature_user_shop_slope'] = result['feature_user_shop_lat_sub'] / result['feature_user_shop_lon_sub']
    return result

# 用户和店铺的方向角
def get_user_shop_degree(result):
    result['feature_user_shop_degree'] = np.arctan2(result['feature_user_shop_lat_sub'],result['feature_user_shop_lon_sub'])
    return result

