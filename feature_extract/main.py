#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import pandas as pd
import datetime
import multiprocessing
import pickle

from .distance import *
from .shop import *
from .other import *
from .wifi import get_wifi_feature1,get_wifi_feature2

# 合并交易表，候选集表，店铺信息
def merge(transation,candidate,shop_info,config,train = True):

    result = transation

    # 加入候选集,训练集有标签,测试集没有标签
    candidate.drop_duplicates(inplace = True)
    if train:
        # result['label'] = 1
        result_positive = result[['row_id','shop_id']]
        candidate_witiout_lcs = candidate[['row_id','shop_id']] # 主要是得到row_id和shop_id
        negative_candidate = candidate_witiout_lcs[~candidate_witiout_lcs.isin(result_positive).all(1)]
        # positive_candidate = candidate_witiout_lcs[candidate_witiout_lcs.isin(result_positive).all(1)]
        result_positive['label'] = 1
        negative_candidate['label'] = 0
        all_candidate = pd.concat([result_positive,negative_candidate],ignore_index=True)

        # del result['shop_id'],result_positive,negative_candidate,positive_candidate
        del result['shop_id'],result_positive,negative_candidate

        result = pd.merge(result,all_candidate, on='row_id', how='left') # 加入候选集里面
        result.label.fillna(0,inplace=True)
        result = pd.merge(result,candidate,on=['row_id','shop_id'],how='left') # 把lcs给加回来
    else:
        result = pd.merge(result,candidate,on='row_id',how='left')

    # # 更改列名
    result.rename(columns = {'match_score':'feature_lj_match_score'},inplace = True)
    result.feature_lj_match_score.fillna(result.feature_lj_match_score.mean(),inplace = True)

    result.rename(columns = {'longitude': 'user_longitude', 'latitude': 'user_latitude'}, inplace = True)
    shop_info.rename(columns = {'longitude':'shop_longitude','latitude':'shop_latitude'},inplace = True)
    if train:
        result = pd.merge(result, shop_info, on=['shop_id','mall_id'], how='left') # 合并表
        result.to_csv(config.train_merge,index = False)
    else:
        result = pd.merge(result, shop_info, on=['shop_id','mall_id'], how='left') # 对于test数据,已经有一个mall_id了
        result.to_csv(config.test_merge,index = False)


def get_features_multiprocess(refer_transation,transation,candidate, shop_info,config,train = True):
    '''
    得到各种特征
    '''

    # 分块获取数据,为了分布式计算
    if train:
        if os.path.exists(config.train_merge):
            result = pd.read_csv(config.train_merge,chunksize=config.chunk_size,iterator=True)
        else:
            merge(transation,candidate,shop_info,config,train = True)
            result = pd.read_csv(config.train_merge,chunksize=config.chunk_size,iterator=True)
    else:
        if os.path.exists(config.test_merge):
            result = pd.read_csv(config.test_merge,chunksize=config.chunk_size,iterator=True)
        else:
            merge(transation,candidate,shop_info,config,train = False)
            result = pd.read_csv(config.test_merge,chunksize=config.chunk_size,iterator=True)

    pool = multiprocessing.Pool(processes=config.pool_size)
    result_all = pd.DataFrame({})
    result_all_list = []
    index = 0
    for result_chunk in result:
        result_all_list.append(pool.apply_async(get_features,(refer_transation,shop_info,result_chunk,config,index,train,)))
        index += 1
    pool.close()
    pool.join()
    for i in result_all_list:
        result_all = pd.concat([result_all,i.get()],axis=0,ignore_index=True)
    print('[所有特征]模型保存成功,多进程结束')

    return result_all



def get_features(refer_transation,shop_info,result,config,index,train = True):
    # 加入候选集,训练集有标签,测试集没有标签

    # print('[特征{}]抽取wifi强度的特征'.format(index))
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # result = get_wifi_feature1(result)
    # if train:
    #     result.to_csv('{}train_feature1.csv'.format(config.cache_dir)) # 用于暂时保存,发生错误时可以读取
    # else:
    #     result.to_csv('{}test_feature1.csv'.format(config.cache_dir))
    #
    #
    # print('[特征{}]抽取用户wifi和店铺wifi匹配程度的特征'.format(index))
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # result = get_wifi_feature2(result,config,train)
    # if train:
    #     result.to_csv('{}train_feature2.csv'.format(config.cache_dir))
    # else:
    #     result.to_csv('{}test_feature2.csv'.format(config.cache_dir))


    print('[特征{}]抽取小时、工作日特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_time_feature(result)
    if train:
        result.to_csv('{}train_feature3.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature3.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户-店铺的距离特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_shop_distance(result)
    if train:
        result.to_csv('{}train_feature4.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature4.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户-店铺平均位置的距离特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_shop_average_distance(refer_transation,result)
    if train:
        result.to_csv('{}train_feature5.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature5.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户-店铺斜率特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_shop_slope(result)
    if train:
        result.to_csv('{}train_feature6.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature6.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户-店铺方向角特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_shop_degree(result)
    if train:
        result.to_csv('{}train_feature7.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature7.csv'.format(config.cache_dir))


    print('[特征{}]抽取店铺热度特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_shop_heat_degree(refer_transation,result)
    if train:
        result.to_csv('{}train_feature8.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature8.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户总访问次数特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_visit_times(refer_transation,result) # 暂时存疑吧
    if train:
        result.to_csv('{}train_feature9.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature9.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户访问该店铺次数/比例特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_visit_shop_times(refer_transation,result)
    if train:
        result.to_csv('{}train_feature10.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature10.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户访问该类别店铺次数/比例特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_visit_shop_category_times(refer_transation,shop_info,result)
    if train:
        result.to_csv('{}train_feature11.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature11.csv'.format(config.cache_dir))


    print('[特征{}]抽取用户平均花费和该店铺消费的差值比例特征'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    result = get_user_shop_cost_sub(refer_transation,shop_info,result)
    if train:
        result.to_csv('{}train_feature12.csv'.format(config.cache_dir))
    else:
        result.to_csv('{}test_feature12.csv'.format(config.cache_dir))

    print('[特征{}]抽取完毕!'.format(index))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))


    return result

