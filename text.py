# -*-coding:utf-8-*-
import os
from generate_accept_sentence_new import generate_main
from utilize.dirpath import text, generic_save_path, target_save_path
import torch
import matplotlib.pyplot as plt

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#generate_main(target_save_path, generic_save_path, device, text, 3000)



# extract sky
# import pandas as pd
# from tqdm import tqdm
#
# NEWFILE = "./data/private/skytrax/airline.txt"
# airline_data = pd.read_csv("./data/private/skytrax/airline.csv")
# airport_data = pd.read_csv("./data/private/skytrax/airport.csv")
# lounge_data = pd.read_csv("./data/private/skytrax/lounge.csv")
# seat_data = pd.read_csv("./data/private/skytrax/seat.csv")
#
#
# f = open(NEWFILE, 'a')
#
# data = [airline_data, airport_data, lounge_data, seat_data]
# names = ["Airline", "Airport", "Lounge", "Seat"]
#
# for c in data[0].columns:
#     print("(%s) %s: %d" % (data[0][c].dtype, c, data[0][c].count()))
#
# single_string = ''
#
# for idx, index in tqdm(enumerate(airline_data['airline_name'].index)):
#     airline_name = str(airline_data['airline_name'].get(index))
#     author = str(airline_data['author'].get(index))
#     country = str(airline_data['author_country'].get(index))
#     date = str(airline_data['date'].get(index))
#     content = str(airline_data['content'].get(index))
#     if country != '':
#         single_string = author + ' is an ' + country + ' who flew on ' + airline_name + ' in ' + date + ' and says ' + content
#     else:
#         single_string = author + ' flew on ' + airline_name + ' in ' + date + ' and says ' + content
#     f.write(single_string)
#     f.write('\n')
#     single_string = ''
# 本文选用BIO标注法，其中”B“表示实体起始位置，
# ”I“表示实体内容位置，”O“表示非实体。将7万条数据样本经过清洗后，
# 按字进行分割，使用BIO标注形式标注四类命名实体，包括人名（PERSON）、
# 地名（LOCATION）、组织机构名（ORGANIAZATION）以及时间（TIME）

# name_list = ['iter_1', 'iter_2', 'iter_3', 'iter_4', 'iter_5']
#
# one = [93, 48, 16, 11, 11]
# two = [78, 38, 17, 38, 8]
# three = [10, 6, 1, 34, 1]
#
# list_2 = []
# for i in range(len(one)):
#     new_value=one[i]+two[i]
#     list_2.append(new_value)
#
# list_3 = []
# for i in range(len(one)):
#     new_value=one[i]+two[i]+three[i]
#     list_3.append(new_value)
#
# x = list(range(len(one)))
# total_width, n = 0.8, 3
# width = total_width / n
#
# plt.bar(x, one, width=width, label='1')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, list_2, width=width, label='2', tick_label=name_list)
#
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, list_3, width=width, label='3', tick_label=name_list)
#
# plt.legend()
# plt.show()


# import pandas as pd
# from tqdm import tqdm
#
# NEWFILE = "./data/private/Rotten tomatoes movie review/review.txt"
# airline_data = pd.read_csv("./data/private/Rotten tomatoes movie review/rotten_tomatoes_critic_reviews.csv")
#
#
#
# f = open(NEWFILE, 'a')
#
# data = [airline_data]
# names = ["critic_name", "publisher_name", "review_date", "review_content"]
#
# for c in data[0].columns:
#     print("(%s) %s: %d" % (data[0][c].dtype, c, data[0][c].count()))
#
# single_string = ''
# for idx, index in tqdm(enumerate(airline_data['critic_name'].index)):
#     critic_name = str(airline_data['critic_name'].get(index))
#     publisher_name = str(airline_data['publisher_name'].get(index))
#     review_date = str(airline_data['review_date'].get(index))
#     review_content = str(airline_data['review_content'].get(index))
#
#     single_string = 'Published on ' + publisher_name + ' in ' + review_date + '.' + critic_name + ' says ' + review_content
#     f.write(single_string)
#     f.write('\n')
#     single_string = ''

# from utilize.dirpath import iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, \
#     iter_5_save_path
#
# from utilize.dirpath import name_save_path_iter1, name_save_path_iter2, name_save_path_iter3, \
#     name_save_path_iter4, name_save_path_iter5
# from sentence_reconstruct import reconstruct_main
# from bertNer import extract_name
# from utilize.dirpath import target_save_path_movie
# from utilize.dirpath import reconstruct_save_path_iter1,\
#                             reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
#                             reconstruct_save_path_iter4, reconstruct_save_path_iter5
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
# save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path]
# save_path = [path + '/movie' for path in save_path]
#
# name_save_path = [name_save_path_iter1, name_save_path_iter2, name_save_path_iter3, name_save_path_iter4,
#                   name_save_path_iter5]
# name_save_path = [path + '/movie' for path in name_save_path]
#
# reconstruct_save_path = [reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,
#                          reconstruct_save_path_iter4, reconstruct_save_path_iter5]
# reconstruct_save_path = [path + '/movie' for path in reconstruct_save_path]
#
#
# for i in range(len(save_path)):
#
#     if not os.path.exists(save_path[i]):
#         os.makedirs(save_path[i])
#
#     if not os.path.exists(reconstruct_save_path[i]):
#         os.makedirs(reconstruct_save_path[i])
#
#     if not os.path.exists(name_save_path[i]):
#         os.makedirs(name_save_path[i])
#
#     extract_name(save_path[i], name_save_path[i])
#     reconstruct_main(device, target_save_path_movie, name_save_path[i], reconstruct_save_path[i])



# import pandas as pd
# from tqdm import tqdm
# from collections import Counter
# import os
# from utilize.dirpath import reconstruct_eval, reconstruct_save_path
#
#
def find_lcseque(s1, s2):
    #  生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    #  d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'

    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)
#
# reconstruct_save_path1 = [reconstruct_save_path + '/recontruct0.csv',
#                          reconstruct_save_path + '/recontruct1.csv',
#                          reconstruct_save_path + '/recontruct2.csv',
#                          reconstruct_save_path + '/recontruct3.csv',
#                          reconstruct_save_path + '/recontruct4.csv']
#
# compare = []
# for i in range(len(reconstruct_save_path1)):
#     airline_data_private = pd.read_csv("./data/private/skytrax/airline.csv")
#     airline_data_reconstruct = pd.read_csv(reconstruct_save_path1[i])
#
#     record = []
#     with open(reconstruct_eval + '/eval_sky' + str(i) + '.txt', 'a+') as f:
#         for idx, index in tqdm(enumerate(airline_data_reconstruct['name'].index)):
#             name = str(airline_data_reconstruct['name'].get(index))
#             airline = str(airline_data_reconstruct['airline'].get(index))
#             country = str(airline_data_reconstruct['country'].get(index))
#             date = str(airline_data_reconstruct['date'].get(index))
#             counts = []
#             info = []
#             if name not in compare:
#                 for idx, index in enumerate(airline_data_private['airline_name'].index):
#                     count = 0
#                     information = ''
#                     name_private = str(airline_data_private['author'].get(index))
#                     airline_private = str(airline_data_private['airline_name'].get(index))
#                     country_private = str(airline_data_private['author_country'].get(index))
#                     date_private = str(airline_data_private['date'].get(index))
#                     #s_name = find_lcseque(name, name_private)
#                     s_airline = find_lcseque(airline, airline_private)
#                     #if (len(s_name) / len(name)) > 0.75 and (len(s_name) / len(name_private)) > 0.75:
#                     if name == name_private:
#                         information = information + name + '. ' + name_private
#                         count = count + 1
#                         if (len(s_airline) / len(airline)) > 0.75 and (len(s_airline) / len(airline_private)) > 0.75:
#                             information = information + ', ' + airline
#                             count = count + 1
#                         if country == country_private:
#                             information = information + ', ' + country
#                             count = count + 1
#                         if date == date_private:
#                             information = information + ', ' + date
#                             count = count + 1
#
#                         # counts.append(count)
#                         # info.append(information)
#                         f.write(information)
#                         f.write('\n')
#                         record.append(count)
#                     if idx == 10000:
#                         # try:
#                         #     f.write(info[counts.index(max(counts))])
#                         #     f.write('\n')
#                         #     record.append(max(counts))
#                         # except ValueError:
#                         #     break
#                         break
#                 compare.append(name)
#         num_Count = Counter(record)
#
#         print(num_Count)

import pandas as pd
from tqdm import tqdm
from collections import Counter
import csv

def sky_eval_combine(reconstruct_save_path):
    df = pd.DataFrame()
    df.columns = ["name", "country", "airline", "date"]
    compare = []
    for i in range(len(reconstruct_save_path)):
        airline_data_private = pd.read_csv("./data/private/skytrax/airline.csv")
        airline_data_reconstruct = pd.read_csv(reconstruct_save_path[i] + '/reconstruct.csv')

        for idx, index in tqdm(enumerate(airline_data_reconstruct['name'].index)):
            name = str(airline_data_reconstruct['name'].get(index))
            airline = str(airline_data_reconstruct['airline'].get(index))
            country = str(airline_data_reconstruct['country'].get(index))
            date = str(airline_data_reconstruct['date'].get(index))
            for idx, index in enumerate(airline_data_private['airline_name'].index):
                name_private = str(airline_data_private['author'].get(index))
                airline_private = str(airline_data_private['airline_name'].get(index))
                country_private = str(airline_data_private['author_country'].get(index))
                date_private = str(airline_data_private['date'].get(index))
                s_airline = find_lcseque(airline, airline_private)
                diction = {
                           'name': 'Nan',
                           'airline': 'Nan',
                            'country': 'Nan',
                            'date': 'Nan'
                        }
                if name == name_private:
                    if name not in compare:
                        diction['name'] = name
                        if (len(s_airline) / len(airline)) > 0.75 and (len(s_airline) / len(airline_private)) > 0.75:
                            diction['airline'] = airline
                        if country == country_private:
                            diction['country'] = country
                        if date == date_private:
                            diction['date'] = date

                        df.append(diction, ignore_index=True)
                    else:
                        change = False
                        index = df[df['name'] == name].index.tolist()[0]
                        for j in range(len(index)):
                            airline_ = str(airline_data_private['airline_name'].get(j))
                            country_ = str(airline_data_private['author_country'].get(j))
                            date_ = str(airline_data_private['date'].get(j))
                            if (len(s_airline) / len(airline)) > 0.75 and (len(s_airline) / len(airline_private)) > 0.75:
                                if airline_ == 'Nan':
                                    df['airline'].loc[j] = airline
                                    change = True
                            if country == country_private:
                                if country_ == 'Nan':
                                    df['country'].loc[j] = country
                                    change = True
                            if date == date_private:
                                if date_ == 'Nan':
                                    df['date'].loc[j] = date
                                    change = True

                            if change:
                                break

                if idx == 10000:
                    break
            compare.append(name)

    df.to_csv(reconstruct_save_path[0] + '/eval_reconstruct.csv')

import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utilize.dirpath import target_save_path_sky, iter_1_save_path, generic_save_path
from tqdm import tqdm
from torch.autograd import Variable
import os
from torch import nn
import numpy as np
import scipy.stats
from utilize.dirpath import reconstruct_save_path_iter1,\
                            reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
                            reconstruct_save_path_iter4, reconstruct_save_path_iter5
import pandas as pd
from collections import Counter
import re
import csv

def compute_sensetive_KL(P1, P2):
    kL = scipy.stats.entropy(P1, P2)
    sensetive = "{0:.4f}".format(kL)
    return sensetive

def read_data(dir):

    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            dataset.append(line)

    return dataset

def find_text(predictions, index_text):
    softmax = nn.Softmax(dim=0)

    prediction_k = predictions[0, -1, :].sort(descending=True)[1]
    probablitys = predictions[0, -1, :].sort(descending=True)[0]

    probablitys = softmax(probablitys)

    a_t2n = prediction_k.cpu().numpy()

    index = np.argwhere(a_t2n == index_text)

    probability = probablitys[index[0][0]]

    return probability.item()


def generate_sentence_probablity(model, device, tokenizer, generate_text):
    model.to(device)
    model.eval()

    total_predicted_text = '<|endoftext|>'

    probabilities = []

    str_list = generate_text.split()

    indexed_tokens = tokenizer.encode(total_predicted_text)

    tokens_tensor = torch.tensor([indexed_tokens])


    for w in str_list:

        tokens_tensor = Variable(tokens_tensor).to(device)

        total_predicted_text = total_predicted_text + ' ' + w

        index = tokenizer.encode(total_predicted_text)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        probability = find_text(predictions, index[-1])
        probabilities.append(probability)

        # total_predicted_text += tokenizer.decode(index[-1])
        #
        # print(tokenizer.decode(index[-1]))
        #
        # print(total_predicted_text)

        indexed_tokens += [index[-1]]
        tokens_tensor = torch.tensor([indexed_tokens])

    return probabilities

def generate_sensetive(target_save_path, generic_save_path, device, save_path):

    tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
    model_target = GPT2LMHeadModel.from_pretrained(target_save_path)

    tokenizer_generic = GPT2Tokenizer.from_pretrained(generic_save_path)
    model_generic = GPT2LMHeadModel.from_pretrained(generic_save_path)


    for j in range(len(save_path)):
        if not os.path.exists(save_path[j]):
            os.makedirs(save_path[j])
        with open(save_path[j] + '/train_sensetive.txt', 'a+') as f:
            text = read_data(save_path[j] + '/train' + str(j+1) + '.txt')
            for i in tqdm(range(len(text))):

                probability_generic = generate_sentence_probablity(model_generic, device, tokenizer_generic, text[i])
                probability_target = generate_sentence_probablity(model_target, device, tokenizer_target, text[i])

                sens_res = compute_sensetive_KL(probability_generic, probability_target)
                f.write(text[i] + ',' + sens_res + '\n')

# PATHS = ['./data/private', './data/private', './data/private','./data/private', './data/private']
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# generate_sensetive(target_save_path, generic_save_path, device, PATHS)

# import zlib
#
#
# def gzip(data, truncate=True):
#     '''
#     基于zlib压缩比进行信息熵分析。
#     这比shannon分析更快，但是不是很准确。
#     '''
#     # 求信息熵的方法：zlib压缩大小/原始大小
#     e = float(float(len(zlib.compress(data.encode('utf-8'), 9))) / float(len(data)))
#
#     if truncate and e > 1.0:
#         e = 1.0
#     return e
#
#
# from utilize.dirpath import target_save_path_sky, iter_1_save_path, generic_save_path, \
#                             iter_2_save_path, iter_3_save_path, iter_4_save_path, \
#                             iter_5_save_path
# from generate_accept_sentence_new import generate_main
# from extract_iter_traindata import extract_accept
# from train_iter_model import train_and_save
# from sentence_reconstruct import reconstruct_main
# from bertNer import extract_name
# from utilize.dirpath import reconstruct_save_path_iter1,\
#                             reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
#                             reconstruct_save_path_iter4, reconstruct_save_path_iter5
#
# from utilize.dirpath import name_save_path_iter1, name_save_path_iter2, name_save_path_iter3, \
#                             name_save_path_iter4, name_save_path_iter5
#
#
# save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path]
# save_path = [path + '/sky_compare' for path in save_path]
#
# reconstruct_save_path = [reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,
#                          reconstruct_save_path_iter4, reconstruct_save_path_iter5]
# reconstruct_save_path = [path + '/sky_compare/throld_4.0595' for path in reconstruct_save_path]
#
# name_save_path = [name_save_path_iter1, name_save_path_iter2, name_save_path_iter3, name_save_path_iter4,
#                   name_save_path_iter5]
#
# name_save_path = [path + '/sky_compare/throld_4.0595' for path in name_save_path]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for i in range(len(save_path)):
#     if not os.path.exists(save_path[i]):
#         os.makedirs(save_path[i])
#
#     if not os.path.exists(reconstruct_save_path[i]):
#         os.makedirs(reconstruct_save_path[i])
#
#     if not os.path.exists(name_save_path[i]):
#         os.makedirs(name_save_path[i])
#
#     extract_accept(save_path[i], 1, 2)
#     extract_name(save_path[i], name_save_path[i])
#     reconstruct_main(device, target_save_path_sky, name_save_path[i], reconstruct_save_path[i])


def mfun2(s1, s2):
    strs = [s1, s2]

    ans = ''
    lens = 0
    for i in zip(*strs):
        if len(set(i)) == 1:

            ans += i[0]
            lens = lens + 1
        else:
            break
    return lens


def find_lcsubstr(s1, s2):
    # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p]

def compute_sum(list):

    list1 = list.copy()
    for i in range(len(list)):
        list[i] = sum(list1[:i+1])
    print(list)
    return list

def fsocre(recall, precision):

    if precision == 0 or recall == 0:
        return 0

    return 2 / (1 / precision + 1 / recall)

our = []

our_len = [204, 327, 420, 473, 493, 569, 585, 498, 595, 558]

total = 85292

our = compute_sum(our)
our_len = compute_sum(our_len)

recall_our = []

recall_our = compute_sum(recall_our)

precision_our = []

f1_our = []


for i in range(len(our)):
    precision_our.append(round(our[i] / our_len[i], 4))
    f1_our.append(round(fsocre(recall_our[i], precision_our[i]), 4))

print("precision_our:", precision_our)

print('f1_our:', f1_our)


