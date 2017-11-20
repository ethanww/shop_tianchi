#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import datetime

def get_transaction(path,train = True):
    df = pd.read_csv(path)
    df['time_stamp']=pd.to_datetime(df['time_stamp']) # 处理成标准时间
    # if train:
    #     df['row_id'] = pd.Series(np.arange(df.shape[0])) # 测试集中已经有row_id
    return df


def list_dir(path):
    result = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        result.append(file_path)
    return result



def euclidean_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)
    dy = np.abs(lat1 - lat2)
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5

    return L

def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h

def manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_distance(lat1, lng1, lat1, lng2)
    b = haversine_distance(lat1, lng1, lat2, lng1)

    return a + b


def rank(data, feat1, feat2, ascending):
    data.sort_values(by=[feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 整理结果
def reshape(pred):
    result = pred.copy()
    result = rank(result,'row_id','pred',ascending=False) #
    result = result[result['rank']<1][['row_id','shop_id','rank']]
    result = result.set_index(['row_id','rank']).unstack()
    result.reset_index(inplace=True)
    result['row_id'] = result['row_id'].astype('int')
    result.columns = ['row_id', 'shop_id']
    return result


# 预测结果
def predict(data,features,model):
    data.loc[:,'pred'] = model.predict(data[features])
    result = reshape(data)
    return result

# 整理结果
def train_reshape(pred):
    result = pred.copy()
    result = rank(result,'row_id','pred',ascending=False) #
    result = result[result['rank']<1][['row_id','shop_id']]
    return result

def train_predict(data,features,model):
    data.loc[:,'pred'] = model.predict(data[features])
    result = train_reshape(data)
    return result
# 获取真实标签,仅用于训练集
def get_label(data):
    result = data[data['label'] == 1][['row_id','shop_id']]
    return result

# 获取分数
def get_score(data,features,model,config):
    result = predict(data,features,model)

    # now = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
    # result_path = '{}{}_train_result_{}.csv'.format(config.result_dir, 'lgb', now)
    # result.to_csv(result_path)
    # print(now)
    # print('保存模型至{}'.format(result_path))

    result_label = get_label(data)

    result.rename(columns = {'shop_id':'predict_shop_id'},inplace = True)
    result = pd.merge(result_label,result,on='row_id',how='left')

    acc = sum(result['predict_shop_id'] == result['shop_id'])
    return acc/result.shape[0]




