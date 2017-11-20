#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''配置项'''

class Config(object):
    def __init__(self):
        self.data_dir = './data/'
        self.data_dir2 = './data2/' # 俊哥的数据...
        self.candidate_dir = './candidate/'
        self.cache_dir = './cache/'
        self.model_dir= './model/'
        self.result_dir = './result/'
        self.time = '2017-11-17'  # 特征保存日期，可以认为是版本号

        self.time_threshod = '2017-08-16 00:00:00' # 训练候选集划分

        # self.train_transaction = self.data_dir + 'ccf_first_round_user_shop_behavior.csv'
        # self.test_transaction = self.data_dir + 'evaluation_public.csv'
        self.train_transaction = self.data_dir2 + 'raw_data_cleaned_combined.csv'
        self.test_transaction = self.data_dir2 + 'test_data_cleaned_combined.csv'
        self.shop_info = self.data_dir + 'ccf_first_round_shop_info.csv'

        # self.train_candidate = self.candidate_dir + 'candidate_train_10.csv'
        # self.test_candidate = self.candidate_dir + 'candidate_test_10.csv'
        self.train_candidate = self.data_dir2 + 'candidate_train_10_rank.csv'
        self.test_candidate = self.data_dir2 + 'candidate_test_10_rank.csv'

        self.pre_shop_wifi = self.data_dir + 'pre_two_weeks_shop_wifi_sorted.csv'
        self.after_shop_wifi = self.data_dir + 'after_two_weeks_shop_wifi_sorted.csv'

        self.train_merge = '{}train_merge.csv'.format(self.data_dir2) # 存储合并了交易表,店铺表,候选集的位置
        self.test_merge = '{}test_merge.csv'.format(self.data_dir2)

        self.train_feature_path = self.cache_dir + 'train_feature{}.csv'.format(self.time)  # 特征保存位置
        self.test_feature_path = self.cache_dir + 'test_feature{}.csv'.format(self.time)

        self.chunk_size = 100000  # chunk大小

        self.pool_size = 30 # 多进程数量

        # lgb参数
        self.lgb_leaves = 96
        self.lgb_lr = 0.05
        self.lgb_boost_round = 2000
        self.lgb_early_stopping_rounds = 40
        self.verbose_eval = 20     # 20轮出一次参数


