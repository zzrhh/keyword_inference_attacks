# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import collections
from utilize.dirpath import iter_2_save_path, iter_1_save_path, \
                            iter_3_save_path, iter_4_save_path, iter_5_save_path, \
                            iter_6_save_path, iter_7_save_path, iter_8_save_path, \
                            iter_9_save_path, iter_10_save_path

from utilize.dirpath import iter_1_save_path_new, iter_2_save_path_new, iter_3_save_path_new, \
                            iter_4_save_path_new, iter_5_save_path_new

from utilize.dirpath import iter_1_save_path_privacy, iter_2_save_path_privacy, \
                            iter_3_save_path_privacy, iter_4_save_path_privacy, \
                            iter_5_save_path_privacy

from utilize.dirpath import iter_1_save_path_compare, iter_2_save_path_compare, \
                            iter_3_save_path_compare, iter_4_save_path_compare, \
                            iter_5_save_path_compare


from utilize.dirpath import text

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


# Path = [iter_1_save_path + '/new/combine/new/all.txt', iter_2_save_path + '/new/combine/new/all.txt',
#          iter_3_save_path + '/new/combine/new/all.txt', iter_4_save_path + '/new/combine/new/all.txt',
#          iter_5_save_path + '/new/combine/new/all.txt', iter_6_save_path + '/all.txt',
#          iter_7_save_path + '/all.txt', iter_8_save_path + '/all.txt',
#          iter_9_save_path + '/all.txt', iter_10_save_path + '/all.txt']

#Path = [iter_1_save_path_new, iter_2_save_path_new, iter_3_save_path_new, iter_4_save_path_new, iter_5_save_path_new]

# Path = [iter_1_save_path_compare, iter_2_save_path_compare, iter_3_save_path_compare,
#         iter_4_save_path_compare, iter_5_save_path_compare]


# Path = [iter_1_save_path_privacy, iter_2_save_path_privacy, iter_3_save_path_privacy,
#         iter_4_save_path_privacy, iter_5_save_path_privacy]

# Path = [path + '/all.txt' for path in Path]
# Path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path]
#
# Path = [path + '/new/combine/new/all.txt' for path in Path]


# Path = ['./data/private/train1sensetive.txt', './data/private/train2sensetive.txt',
#         './data/private/train3sensetive.txt', './data/private/train4sensetive.txt',
#         './data/private/train5sensetive.txt']

# save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path]
# save_path = [path + '/throld_5.7535/all.txt' for path in save_path]

save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path]
save_path = [path + '/medical/k_20/limit_1/our_method/all.txt' for path in save_path]


def compute_th(dir):

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

    throld = mu + sigma
    print(throld)

    return throld


# for i in range(len(Path)):
#     th = compute_th(Path[i])
#     print(th)



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

    # plt.xlim((0, 2))
    # plt.ylim((0, 6.0))
    #
    # my_x_ticks = np.arange(0, 2, 0.2)
    # my_y_ticks = np.arange(0, 6.0, 0.5)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    plt.show()

def paint_compare(sen1,sen2):
    b = collections.Counter(sen1)
    b1 = collections.Counter(sen2)

    dic = {number: value for number, value in b.items()}
    dic1 = {number: value for number, value in b1.items()}

    x = [float(i) for i in dic.keys()]
    x1 = [float(i) for i in dic1.keys()]

    y = []
    y1 = []
    # 取得value
    for i in dic.keys():
        y.append(dic.get(i))

    for i in dic1.keys():
        y1.append(dic1.get(i))

    # 求均值
    mu = np.mean(x)
    mu1 = np.mean(x1)
    # 求总体标准差
    sigma = np.std(x)
    sigma1 = np.std(x1)

    num_bins = 200  # 直方图柱子的数量

    plt.figure()
    n, bins, patches = plt.hist(x, num_bins, density=1, stacked=True)

    n1, bins1, patches1 = plt.hist(x1, num_bins, density=1, stacked=True)

    plt.figure(dpi=300, figsize=(10, 6))

    y = norm.pdf(bins, mu, sigma)
    y1 = norm.pdf(bins1, mu1, sigma1)

    plt.figure(dpi=300, figsize=(10, 8))

    plt.plot(bins, y, label='Filter Dataset', color='orange', linestyle='-', ms=12, linewidth=3)
    plt.plot(bins1, y1, label='Target Dataset', color='blue', linestyle='--', ms=12, linewidth=3)


    min_distance = 1
    plot_index = 0
    for i in range(len(bins)):
        for j in range(len(bins1)):
            distance = (bins[i] - bins1[j])**2 + (y[i] - y1[j])**2
            if distance < min_distance:
                min_distance = distance
                plot_index = i


    plt.plot(bins[plot_index], y[plot_index], color='red', marker='*', ms=18)

    #plt.plot(sigma + mu, y[plot_index], color='green', marker='*', ms=18)

    #plt.axvline(bins[plot_index], linestyle='--', linewidth=3, ymax=y[plot_index])
    plt.axvline(sigma + mu, linestyle='--', linewidth=3, ymax=y1[plot_index] + 0.35, color='black')
    plt.axvline(sigma * 2 + mu, linestyle='--', linewidth=3, ymax=y1[plot_index] + 0.35, color='black')
    print("plot_x:", bins[plot_index])

    #print('throld:', sigma + mu)
    plt.xlim((0, 2.5))
    plt.ylim((0, 3))
    my_x_ticks = np.arange(0, 2.5, 0.5)
    my_y_ticks = np.arange(0, 3, 0.5)

    plt.xticks(my_x_ticks, fontsize=22, fontweight='medium')
    plt.yticks(my_y_ticks, fontsize=22, fontweight='medium')
    # plt.ylabel(ylabel, fontsize=26, fontweight='medium')
    # plt.xlabel('Iteration', fontsize=26, fontweight='medium')
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig('./data/figs/throld.pdf', bbox_inches='tight')
    plt.show()


savepath = './data/private/train_sensetive.txt'
# for i in range(len(save_path)):
#     res = load_sensetive(save_path[i])
#     paint_throld(res)


res1 = load_sensetive(save_path[0])

res2 = load_sensetive(savepath)

paint_compare(res1, res2)


