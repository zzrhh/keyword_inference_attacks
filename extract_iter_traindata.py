# -*-coding:utf-8-*-
from utilize.dirpath import target_save_path, iter_1_save_path, generic_save_path, \
                            iter_2_save_path, iter_3_save_path, iter_4_save_path, \
                            iter_5_save_path
from compute_throld import compute_th
import math

'''
path = iter_1_save_path
'''

def read_data(dir):
    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            sens = line[-7:].strip(',')
            sens = sens.strip('\n')
            text = line[:-7].strip(',')
            try:
                sens = float(sens)
            except ValueError:
                sens = line[-4:].strip(',')
                text = line[:-4].strip(',')
                try:
                    sens = float(sens)
                except ValueError:
                    text = line
            dataset.append(text)
    return dataset


def load_sensetive(dir):
    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line[-7:].strip(',')
            line = line.strip('\n')
            try:
               line = float(line)
            except ValueError:
                line = line[-4:].strip(',')
                try:
                    line = float(line)
                except ValueError:
                    line = 0.0
            dataset.append(line)
    return dataset


def extract_accept(path, limit, path_all):
    throld = compute_th(path_all[0] + '/all.txt', limit)
    sen = load_sensetive(path + '/all.txt')
    data = read_data(path + '/all.txt')

    with open(path + '/accept.txt', 'w+') as f:
        for i in range(len(sen)):
            if float(sen[i]) > throld:
                f.write(data[i] + '\n')


import zlib
import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.stats import norm

def gzip(data, truncate=True):

    zlib_entropy = []
    for i in range(len(data)):
        if float(len(data[i])) == 0:
            e = 0
        else:
            e = float(float(len(zlib.compress(data[i].encode('utf-8'), 9))) / float(len(data[i])))

            e = float("{0:.4f}".format(e))
            if truncate and e > 1.0:
                e = 1.0
        zlib_entropy.append(e)
    return zlib_entropy

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def tanh(x):
    sig = 2 / (1 + math.exp(-2 * x)) - 1
    return sig

def normal(data, max, min):
    range = max - min
    data = (data - min) / range
    return float("{0:.4f}".format(data))

def paint_throld(sen):
    b = collections.Counter(sen)

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

    print(mu, sigma)

    num_bins = 200  # 直方图柱子的数量
    n, bins, patches = plt.hist(x, num_bins, density=1, stacked=True)
    y = norm.pdf(bins, mu, sigma)

    plt.plot(bins, y, '--')

    plt.xlim((0, 1))
    plt.ylim((0, 8.0))

    my_x_ticks = np.arange(0, 1, 0.2)
    my_y_ticks = np.arange(0, 8.0, 1.0)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.show()


def compute_throld_gzip(path, limit):
    sen_0 = load_sensetive(path + '/all.txt')
    data_0 = read_data(path + '/all.txt')
    max_s = max(sen_0)
    min_s = min(sen_0)
    max_zip = max(gzip(data_0))
    min_zip = min(gzip(data_0))
    zip = gzip(data_0)
    sen_zip = []
    for i in range(len(sen_0)):
        #sensestive = 0.5 * normal(sen_0[i], max_s, min_s) + 0.5 * normal(zip[i], max_zip, min_zip)
        sensestive = 0.5 * sigmoid(sen_0[i]) + 0.5 * zip[i]
        sen_zip.append(sensestive)
    mu = np.mean(sen_zip)

    sigma = np.std(sen_zip)

    throld = sigma * limit + mu

    return throld


def extract_accept_gzip(path, limit, path_all):

    sen = load_sensetive(path + '/all.txt')
    data = read_data(path + '/all.txt')
    max_s = max(sen)
    min_s = min(sen)
    max_zip = max(gzip(data))
    min_zip = min(gzip(data))
    zip = gzip(data)

    sen_zip = []
    for i in range(len(sen)):
        sensestive = 0.5 * normal(sen[i], max_s, min_s) + 0.5 * zip[i]
        #sensestive = 0.5 * sigmoid(sen[i]) + 0.5 * zip[i]
        sen_zip.append(sensestive)

    paint_throld(sen_zip)

    throld = compute_throld_gzip(path_all[0], limit)

    print(throld)
    for i in range(len(sen_zip)):
        if float(sen_zip[i]) > throld:
            with open(path + '/accept.txt', 'a+') as f:
                f.write(data[i] + '\n')


def extract_zip_accept(path):
    data = read_data(path + '/accept.txt')
    zip = gzip(data)
    mu = np.mean(zip)

    sigma = np.std(zip)

    throld = mu - sigma

    with open(path + '/zip_accept.txt', 'w+') as f:
        for j in range(len(zip)):
            if float(zip[j]) > throld:
                f.write(data[j])


