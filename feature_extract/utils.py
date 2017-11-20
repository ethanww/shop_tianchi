#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

DIVIDE_PROTECTION = 0.01

def inverse_pairs(data):
        if len(data) <= 0:
            return 0
        count = 0
        copy = []
        for i in range(len(data)):
            copy.append(data[i])
        copy.sort()
        i = 0
        while len(copy) > i:
            count += data.index(copy[i])
            data.remove(copy[i])
            i += 1
        return count


class Distance(object):
    '''
    计算相似度
    list1:用户wifi列表
    list2:店铺wifi列表
    注意:店铺wifi列表有可能小于用户wifi列表,即len(list1)>=len(list2)
    '''

    @staticmethod
    def common_set_number_and_ratio(wifi_list1,wifi_list2):
        result_list = [0.0] * 5
        common_wifi_set = set(wifi_list1) & set(wifi_list2)
        all_wifi_set = set(wifi_list1) | set(wifi_list2)
        wifi_dict = {}
        for i in range(len(wifi_list2)):
            wifi_dict[wifi_list2[i]] = i
        common_wifi_index = []
        for j in common_wifi_set:
            common_wifi_index.append(wifi_dict[j])
        # 求逆序对
        result_list[0] = len(common_wifi_set)
        result_list[1] = result_list[0] / (len(wifi_list2) + 0.01)
        result_list[2] = result_list[0] / (len(all_wifi_set) + 0.01)
        result_list[3] = inverse_pairs(common_wifi_index) # 逆序数
        result_list[4] = result_list[3] / (result_list[0] + DIVIDE_PROTECTION)
        return result_list

    @staticmethod
    def lcs_length(wifi_list1, wifi_list2):
        table = [[0] * (len(wifi_list2) + 1) for _ in range(len(wifi_list1) + 1)]
        for i, ca in enumerate(wifi_list1, 1):
            for j, cb in enumerate(wifi_list2, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1 if ca == cb else
                    max(table[i][j - 1], table[i - 1][j]))
        return table[-1][-1]

    @staticmethod
    def manhattan_distance(wifi_list1,list1,wifi_list2,list2):
        distance_list = [0.0]*7
        for i in range(len(wifi_list2)):
            if wifi_list2[i] in wifi_list1:
                distance_list[0] += abs(list1[i]-list2[i]) # 如果在交集中,对应位置相减
            else:
                distance_list[1] += abs(list1[i]-list2[i])

        distance_list[2] = distance_list[0] + distance_list[1] # 总的距离
        distance_list[3] = distance_list[0] + distance_list[1]*2 # 不在交集内的距离加罚
        distance_list[4] = distance_list[0] / (distance_list[1] + DIVIDE_PROTECTION) # 在交集内的距离/不在交集内的距离
        distance_list[5] = distance_list[0] / (distance_list[2] + DIVIDE_PROTECTION) # 在交集内的距离/总的距离
        distance_list[6] = distance_list[0] / (distance_list[3] + DIVIDE_PROTECTION) # 在交集内的距离/总的距离+加罚

        return distance_list

    @staticmethod
    def euclidean_distance(wifi_list1,list1,wifi_list2,list2):
        distance_list = [0.0]*7
        for i in range(len(wifi_list2)):
            if wifi_list2[i] in wifi_list1:
                distance_list[0] += (list1[i]-list2[i])**2 # 如果在交集中,对应位置相减
            else:
                distance_list[1] += (list1[i]-list2[i])**2

        distance_list[2] = distance_list[0] + distance_list[1] # 总的距离
        distance_list[3] = distance_list[0] + distance_list[1]*2 # 不在交集内的距离加罚
        distance_list[4] = distance_list[0] / (distance_list[1] + DIVIDE_PROTECTION) # 在交集内的距离/不在交集内的距离
        distance_list[5] = distance_list[0] / (distance_list[2] + DIVIDE_PROTECTION) # 在交集内的距离/总的距离
        distance_list[6] = distance_list[0] / (distance_list[3] + DIVIDE_PROTECTION) # 在交集内的距离/总的距离+加罚

        return distance_list

    @staticmethod
    def cosine_similarity(list1,list2):
        x = 0.0
        y = 0.0
        xy = 0.0
        for i in range(len(list2)):
            x += list1[i]**2
            y += list2[i]**2
            xy += list1[i] * list2[i]
        return xy/(math.sqrt(x*y)+DIVIDE_PROTECTION)

    @staticmethod
    def match_score(wifi_list1,wifi_list2,list2):
        match_score = 0.0
        for i in range(len(wifi_list2)):
            if wifi_list2[i] in wifi_list1:
                match_score += list2[i]

        return match_score/(sum(list2) + DIVIDE_PROTECTION)
