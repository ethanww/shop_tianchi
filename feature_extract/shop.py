#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime

'''获取店铺相关的特征'''

DIVIDE_PROTECT = 0.01

# 获取店铺热度以及比率
def get_shop_heat_degree(refer,result):
    shop_user_count = refer.groupby(['shop_id','user_id'],as_index = False)['row_id'].agg({'feature_shop_user_count':'count'}) # 该店铺有多少人访问过
    shop_visit_count = refer.groupby('shop_id',as_index = False)['row_id'].agg({'feature_shop_visit_count':'count'}) # 该店铺有多少次访问过
    mall_user_count = refer.groupby(['mall_id','user_id'],as_index = False)['row_id'].agg({'feature_mall_user_count':'count'}) # 该商场有多少人访问过
    mall_visit_count = refer.groupby('mall_id',as_index = False)['row_id'].agg({'feature_mall_visit_count':'count'}) # 该商场有多少人访问过

    result = pd.merge(result,shop_user_count,on=['shop_id','user_id'],how='left')
    result = pd.merge(result,shop_visit_count,on='shop_id',how='left')
    result = pd.merge(result,mall_user_count,on=['mall_id','user_id'],how='left')
    result = pd.merge(result,mall_visit_count,on='mall_id',how='left')
    # 填补缺失值,用均值填补
    result.feature_shop_user_count.fillna(result.feature_shop_user_count.mean(), inplace=True)
    result.feature_shop_visit_count.fillna(result.feature_shop_visit_count.mean(),inplace=True)
    result.feature_mall_user_count.fillna(result.feature_mall_user_count.mean(),inplace=True)
    result.feature_mall_visit_count.fillna(result.feature_mall_visit_count.mean(),inplace=True)

    result['feature_shop_mall_user_heat_ratio'] = result['feature_shop_user_count']/(result['feature_mall_user_count']+DIVIDE_PROTECT)
    result['feature_shop_mall_visit_heat_ratio'] = result['feature_shop_visit_count']/(result['feature_mall_visit_count']+DIVIDE_PROTECT)

    return result

# 用户消费总次数
def get_user_visit_times(refer,result):
    user_visit_count = refer.groupby(['user_id'],as_index = False)['row_id'].agg({'feature_user_visit_count':'count'})
    result = pd.merge(result,user_visit_count,on=['user_id'],how='left')

    result.feature_user_visit_count.fillna(result.feature_user_visit_count.mean(), inplace=True)

    return result

# 用户来过该店铺次数,去过该商铺次数,比率
def get_user_visit_shop_times(refer,result):
    user_shop_count = refer.groupby(['shop_id','user_id'],as_index = False)['row_id'].agg({'feature_user_visit_shop_count':'count'})
    user_mall_count = refer.groupby(['mall_id','user_id'],as_index = False)['row_id'].agg({'feature_user_visit_mall_count':'count'})

    result = pd.merge(result,user_shop_count,on=['shop_id','user_id'],how='left')
    result = pd.merge(result,user_mall_count,on=['mall_id','user_id'],how='left')
    result.feature_user_visit_shop_count.fillna(value=0, inplace=True) # 填补空值
    result.feature_user_visit_mall_count.fillna(value=0, inplace=True)

    result['feature_shop_mall_user_visit_ratio'] = result['feature_user_visit_shop_count']/(result['feature_user_visit_shop_count'] + DIVIDE_PROTECT) # 防止除0
    result['feature_user_shop_visit_ratio'] = result['feature_user_visit_shop_count']/(result['feature_user_visit_count'] + DIVIDE_PROTECT)
    result['feature_user_mall_visit_ratio'] = result['feature_user_visit_mall_count']/(result['feature_user_visit_count'] + DIVIDE_PROTECT)

    return result

# 用户去过该类别店铺多少次(是不是可以把这个类别扩大),去过该类别店铺和总去过次数的对比
def get_user_visit_shop_category_times(refer,shop_info,result):
    refer = pd.merge(refer,shop_info,on=['shop_id','mall_id'],how='left')
    user_shop_category_count = refer.groupby(['category_id','user_id'],as_index = False)['row_id'].agg({'feature_user_visit_shop_category_count':'count'})
    result = pd.merge(result,user_shop_category_count,on=['category_id','user_id'],how='left')

    result.feature_user_visit_shop_category_count.fillna(value=0, inplace=True)
    result['feature_user_visit_category_ratio'] = result['feature_user_visit_shop_category_count']/result['feature_user_visit_count']
    result.feature_user_visit_category_ratio.fillna(value=0, inplace=True)

    del result['feature_user_visit_count']

    return result


# 用户以前消费金额的最大值,最小值,均值,和去过该店铺的price的比较
def get_user_shop_cost_sub(refer,shop_info,result):
    refer = pd.merge(refer,shop_info,on=['shop_id','mall_id'],how='left')
    user_cost_max = refer.groupby('user_id',as_index = False)['price'].agg({'feature_user_cost_max':'max'})
    user_cost_average = refer.groupby('user_id',as_index = False)['price'].agg({'feature_user_cost_average':'mean'})
    user_cost_min = refer.groupby('user_id',as_index = False)['price'].agg({'feature_user_cost_min':'min'})

    result = pd.merge(result,user_cost_max,on='user_id',how='left')
    result = pd.merge(result,user_cost_average,on='user_id',how='left')
    result = pd.merge(result,user_cost_min,on='user_id',how='left')

    # 直接用其他数据的平均值填，可能有问题
    result['feature_user_cost_max'].fillna(result.feature_user_cost_max.mean(),inplace=True)
    result['feature_user_cost_average'].fillna(result.feature_user_cost_average.mean(),inplace=True)
    result['feature_user_cost_min'].fillna(result.feature_user_cost_min.mean(),inplace=True)

    result['feature_user_shop_cost_sub_max'] = result['feature_user_cost_max'] - result['price']
    result['feature_user_shop_cost_sub_average'] = result['feature_user_cost_average'] - result['price']
    result['feature_user_shop_cost_sub_min'] = result['feature_user_cost_min'] - result['price']

    result['feature_user_shop_cost_sub_max_ratio'] = result['feature_user_shop_cost_sub_max'] / result['price']
    result['feature_user_shop_cost_sub_average_ratio'] = result['feature_user_shop_cost_sub_average'] / result['price']
    result['feature_user_shop_cost_sub_min_ratio'] = result['feature_user_shop_cost_sub_min'] / result['price']

    return result
