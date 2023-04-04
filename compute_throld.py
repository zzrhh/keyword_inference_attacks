# -*-coding:utf-8-*-
import numpy as np
import collections
from utilize.dirpath import target_save_path, iter_1_save_path

def load_sensetive(dir):
    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            if ',' in line[-8:]:
                lines = line[-7:]
                lines = lines.strip('\n')
                try:
                    lines = float(lines)
                except ValueError:
                    lines = 0.0
            else:
                lines = 0.0

            if lines > 5:
                print(lines, line)
            dataset.append(lines)
    return dataset


def compute_th(dir, limit):

    res = load_sensetive(dir)

    b = collections.Counter(res)
    dic = {number: value for number, value in b.items()}

    x = [float(i) for i in dic.keys()]

    y = []
    # 取得value
    for i in dic.keys():
        y.append(dic.get(i))

    # 求均值
    mu = np.mean(x)

    # 求总体标准差
    sigma = np.std(x)

    throld = sigma * limit + mu
    print(throld)
    print(mu, sigma)

    return throld


path_new = '/sky_final/our_method'
throld = compute_th(iter_1_save_path + path_new + '/all.txt', 1)



print(throld)

