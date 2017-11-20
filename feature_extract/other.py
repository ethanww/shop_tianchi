#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# 增加时间和是否工作日特征
def get_time_feature(result):
    result['feature_hour'] = pd.to_datetime(result['time_stamp']).dt.hour
    result['day'] = pd.to_datetime(result['time_stamp']).dt.day
    day = result['day'].values
    workday = []
    for i in day:
        if i <6:
            workday.append(1)
        else:
            workday.append(0)
    result.loc[:,'feature_workday'] = workday
    del result['day']
    return result


