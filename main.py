#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm as lgb
import datetime
import fire
import gc
import os
import pickle
from sklearn.model_selection import train_test_split

from config import Config
from utils import get_transaction,get_score,predict,train_predict
from feature_extract import get_features



def train():
    # 获取参数
    opt = Config()

    # # 获取数据
    # print('[数据]获取训练集交易表和候选集')
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # train_transation = get_transaction(opt.train_transaction,train = True) # 增加了一行row_id
    # train_candidate = pd.read_csv(opt.train_candidate)
    #
    # print('[数据]按照时间{}划分训练集的参照集'.format(opt.time_threshod))
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # train_refer = train_transation[train_transation.time_stamp < opt.time_threshod]
    # train_after = train_transation[train_transation.time_stamp >= opt.time_threshod]
    #
    # print('[数据]获取测试集交易表和候选集')
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # test_transation = get_transaction(opt.test_transaction,train = False)
    # test_candidate = pd.read_csv(opt.test_candidate)
    #
    # print('[数据]获取商铺信息')
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # shop_info = pd.read_csv(opt.shop_info)
    #
    # # 构造特征/训练集获取标签
    # print('[数据]获取训练集特征/标签并保存')
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # train_features = get_features(train_refer,train_after,train_candidate,shop_info,opt,train = True)
    # train_features.to_csv(opt.train_feature_path,index = False)
    #
    # print('[数据]获取测试集集特征并保存')
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # test_features = get_features(train_after,test_transation,test_candidate,shop_info,opt,train = False)
    # test_features.to_csv(opt.test_feature_path,index = False)
    #
    # # 保存特征
    # print('[数据]成功保存特征到{}'.format(opt.cache_dir))
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    #
    # # 删除
    # del train_transation,train_candidate,test_transation,test_candidate,shop_info
    # gc.collect()


    # 读取训练集和测试集
    print('[数据]读取训练集特征并融合')
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # train_features1 = pd.read_csv('./cache/train_feature2017-11-17_other.csv')
    # train_features2 = pd.read_csv('./cache/train_feature2017-11-17_wifi.csv')
    # train_features = pd.merge(train_features1,train_features2,on=['row_id','shop_id'],how='left')
    train_features = pd.read_csv('./cache/train_feature2017-11-17.csv')
    print('columns:')
    print(train_features.columns)

    # print('[数据]读取测试集特征并融合')
    # now = datetime.datetime.now()
    # print(now.strftime('%Y-%m-%d %H:%M:%S'))
    # test_features1 = pd.read_csv('./cache/test_feature2017-11-17_part1.csv')
    # test_features2 = pd.read_csv('./cache/test_feature2017-11-17_part2.csv')
    # test_features = pd.merge(test_features1,test_features2,on=['row_id','shop_id'],how='left')
    # test_features.rename(columns = {'feature_lj_match_score_x': 'feature_lj_match_score'}, inplace = True)
    # print('columns:')
    # print(test_features.columns)
    #
    # del test_features1,test_features2
    # gc.collect()

    # #----------------------------------------------准备数据-------------------------------------------------------------#
    #
    # # 划分验证集
    # train,val = train_test_split(train_features, test_size=0.1)
    #
    # # 定义使用的特征
    # except_columns = ['feature_shop_user_count','feature_mall_user_count','feature_mall_visit_count',
    #                   'feature_shop_mall_user_heat_ratio','feature_shop_mall_visit_heat_ratio',
    #                   'feature_user_visit_mall_count','feature_shop_mall_user_visit_ratio',
    #                   'feature_user_shop_visit_ratio','feature_user_mall_visit_ratio',
    #                   'feature_user_visit_shop_category_count','feature_user_visit_category_ratio',
    #                   'feature_user_cost_max','feature_user_cost_average','feature_user_cost_min',
    #                   'feature_wifi_max_score','feature_wifi_min_score','feature_wifi_average_score']
    except_columns = ['feature_wifi_match_score','feature_shop_user_count','feature_shop_visit_count',
                      'feature_mall_user_count','feature_mall_visit_count','feature_user_visit_count',
                      'feature_user_visit_shop_count','feature_user_visit_mall_count',
                      'feature_user_visit_shop_category_count','feature_user_cost_max','feature_user_cost_average',
                      'feature_user_cost_min']
    # except_columns = ['feature_wifi_max_score','feature_wifi_min_score','feature_wifi_average_score',
    #                   'feature_lj_match_score_x','feature_lj_match_score_y']
    columns = [c for c in train_features.columns if c[:7] == 'feature' and c not in except_columns]
    # # columns = [c for c in train_features.columns if c[:7] == 'feature']
    # print('使用特征的维度：{}'.format(len(columns)))
    # print('使用的特征:')
    # print(columns)
    #
    # # 定义数据集
    # x_train = train[columns]
    # y_train = train['label']
    # x_val = val[columns]
    # y_val = val['label']
    # del train,val
    # gc.collect()

    #
    # # #----------------------------------------------模型----------------------------------------------------------------#
    # #
    # lgb_train = lgb.Dataset(x_train,y_train)
    # lgb_val = lgb.Dataset(x_val,y_val)
    #
    # # 参数
    # params = {
	 #    'objective': 'binary',
	 #    'metric': {'auc', 'binary_logloss'},
	 #    'is_unbalance': True,
	 #    'num_leaves': opt.lgb_leaves,
	 #    'learning_rate': opt.lgb_lr,
	 #    'feature_fraction': 0.886, # 随机选一部分特征
	 #    'bagging_fraction': 0.886, # 随机选一部分数据
	 #    'bagging_freq': 3,
    #     'max_depth': 8,
    #     'subsample':0.88
    #     # 'min_data_in_bin':3,
    # }
    #
    # # 训练
    # lgb_model = lgb.train(
    #
    #     params,
    #     lgb_train,
    #     num_boost_round=opt.lgb_boost_round,
    #     valid_sets=[lgb_train,lgb_val],
    #     verbose_eval=opt.verbose_eval,
    #     early_stopping_rounds=opt.lgb_early_stopping_rounds,
    # )
    #
    # # 保存模型
    # try:
    #     now = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
    #     save_path = '{}{}_{}.pkl'.format(opt.model_dir, 'lgb', now)
    #     with open(save_path,'wb') as f:
    #         pickle.dump(lgb_model,f)
    #     print(now)
    #     print('[模型]模型成功保存到{}'.format(save_path))
    # except FileNotFoundError:
    #     print('模型保存失败')
    # gc.collect()
    # del x_train,y_train,x_val,y_val

    # 读取模型
    print('读取模型')
    with open('lgb_2017-11-19#02:30:36.pkl','rb') as f:
        lgb_model = pickle.load(f)
    print('载入模型成功')
    print("Features importance...")
    gain =lgb_model.feature_importance('gain')
    ft = pd.DataFrame({'feature':lgb_model.feature_name(), 'split':lgb_model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print(ft.head(100))



    # # 得到测试集结果
    # test_result = train_predict(test_features,columns,lgb_model)
    # now = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
    # result_path = '{}{}_test_result_{}.csv'.format(opt.result_dir, 'lgb', now)
    # test_result.to_csv(result_path,index = False)
    # print('[训练集]文件保存完成！')

    # 在训练集上看结果
    gc.collect()
    score = get_score(train_features,columns,lgb_model,opt)
    print('[结果]训练集分数:{}'.format(score))





if __name__ == '__main__':
    train()
