import numpy as np
import pandas as pd


def fed_max(max_list):
    if isinstance(max_list[0], pd.Series):
        max_l = [series.max() for series in max_list]
        return max(max_l)
    return max(max_list)


def fed_min(min_list):
    if isinstance(min_list[0], pd.Series):
        min_l = [series.min() for series in min_list]
        return min(min_l)
    return min(min_list)


def fed_mean(x_list):
    sum_list = []
    num_all = 0
    for x in x_list:
        sum_list.append(x.sum())
        num_all += len(x)
    return np.sum(sum_list) / num_all


def fed_std(x_list):
    std_list = []
    num_all = 0
    mu = fed_mean(x_list)
    for x in x_list:
        tmp = np.array(x-mu)
        std_list.append(np.power(tmp, 2).sum())
        num_all += len(x)
    std_2 = np.sum(std_list) / num_all

    return np.sqrt(std_2)
