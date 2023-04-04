# -*-coding:utf-8-*-
from keybert import KeyBERT
import numpy as np
from utilize.dirpath import iter_2_save_path, keywords_save_path, iter_1_save_path, \
                            iter_3_save_path, iter_4_save_path, iter_5_save_path, \
                            iter_6_save_path, iter_7_save_path, iter_8_save_path, \
                            iter_9_save_path, iter_10_save_path, Random_baseline, \
                            LM_baseline_movie, LM_baseline_medical
import os
import matplotlib.pyplot as plt
import torch
from utilize.dirpath import iter_1_save_path_compare, iter_2_save_path_compare, \
                            iter_3_save_path_compare, iter_4_save_path_compare, \
                            iter_5_save_path_compare

from utilize.dirpath import iter_1_save_path_privacy, iter_2_save_path_privacy, \
                            iter_3_save_path_privacy, iter_4_save_path_privacy, \
                            iter_5_save_path_privacy

from utilize.dirpath import iter_1_save_path_new, iter_2_save_path_new, iter_3_save_path_new, \
                            iter_4_save_path_new, iter_5_save_path_new
from tqdm import tqdm


torch.cuda.set_device(3)


if not os.path.exists(keywords_save_path):
    os.makedirs(keywords_save_path)


def delete_repeat(data):
    formatList = []
    for id in data:
        if id not in formatList:
            formatList.append(id)
    return formatList


def not_delete(data):
    return data


def read_data(dir):

    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            if line != '':
                dataset.append(line)
    return dataset


def read_private_data():
    private_data = []
    with open('./data/private/train.txt', 'r') as f:

        for idx, line in enumerate(f):
            li = line.strip()
            lin = str(li).lstrip("BACKGROUND OBJECTIVE METHODS RESULTS CONCLUSIONS")
            lin = lin.strip()
            if lin != '' and lin[0:3] != '###':

                private_data.append(lin.strip('\n'))

            if idx == 100000:
                break

    return private_data


def read_private_movie():
    private_data = []
    with open('./data/private/Rotten tomatoes movie review/movie_critic_reviews.txt', 'r') as f:

        for idx, line in enumerate(f):
            li = line.strip()
            private_data.append(li.strip('\n'))
            if idx == 100000:
                break

    return private_data


def load_sensetive(dir):
    data = np.load(dir).tolist()
    return data


def extract_key_new(top_n,Path,n_range, private_data):    #看提取的关键字有没有出现在句子里边


    #private_data = [x.split(' ') for x in private_data]

    base_key = []
    union = []
    doc_all = []

    # for i in range(len(private_data)):
    #     base_key.extend(private_data[i])

    print(len(private_data))
    union.append(private_data)
    for p in range(len(Path)):
        infer_key = []
        doc = read_data(Path[p])
        doc_all.extend(doc)

        keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=n_range, stop_words='english',
                                             use_mmr=True, diversity=0.8, top_n=top_n)
        for i in range(len(doc)):
            for j in range(top_n):
                try:
                    infer_key.append(keywords[i][j][0])
                except IndexError:
                    pass
                continue

        union.append(infer_key)

    return union


# def extract_key(top_n,Path,n_range):
#
#     private_data = read_private_movie()
#
#     keywords_private = kw_model.extract_keywords(private_data, keyphrase_ngram_range = n_range, stop_words='english',
#                                                  use_mmr=True, diversity=1, top_n=top_n)
#
#     base_key = []
#     union = []
#     doc_all = []
#
#     for i in range(len(private_data)):
#         for j in range(top_n):
#             try:
#                 base_key.append(keywords_private[i][j][0])
#             except IndexError:
#                 pass
#             continue
#
#     print(len(base_key))
#     union.append(base_key)
#     for p in range(len(Path)):
#         infer_key = []
#         doc = read_data(Path[p])
#         doc_all.extend(doc)
#
#         keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=n_range, stop_words='english',
#                             use_mmr=True, diversity=0.8, top_n=top_n)
#         for i in range(len(doc)):
#             for j in range(top_n):
#                 try:
#                     infer_key.append(keywords[i][j][0])
#                 except IndexError:
#                     pass
#                 continue
#
#         union.append(infer_key)
#
#     '''
#     infer_key = []
#     keywords = kw_model.extract_keywords(doc_all, keyphrase_ngram_range=n_range, stop_words='english',
#                                          use_mmr=True, diversity=0.8, top_n=top_n)
#     for i in range(len(doc_all)):
#         for j in range(top_n):
#             try:
#                 infer_key.append(keywords[i][j][0])
#             except IndexError:
#                 pass
#             continue
#
#     union.append(infer_key)
#     '''
#
#     return union

def fsocre(recall, precision):
    return 2 / (1 / precision + 1 / recall)


# def find_lcseque(s1, s2):
#     #  生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
#     m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
#     #  d用来记录转移方向
#     d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
#
#     for p1 in range(len(s1)):
#         for p2 in range(len(s2)):
#             if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
#                 m[p1 + 1][p2 + 1] = m[p1][p2] + 1
#                 d[p1 + 1][p2 + 1] = 'ok'
#             elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
#                 m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
#                 d[p1 + 1][p2 + 1] = 'left'
#             else:  # 上值大于左值，则该位置的值为上值，并标记方向up
#                 m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
#                 d[p1 + 1][p2 + 1] = 'up'
#
#     (p1, p2) = (len(s1), len(s2))
#     s = []
#     while m[p1][p2]:  # 不为None时
#         c = d[p1][p2]
#         if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
#             s.append(s1[p1 - 1])
#             p1 -= 1
#             p2 -= 1
#         if c == 'left':  # 根据标记，向左找下一个
#             p2 -= 1
#         if c == 'up':  # 根据标记，向上找下一个
#             p1 -= 1
#     s.reverse()
#     return ''.join(s)

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
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0   # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    return s1[p-mmax:p]

def get_whole_accurcy_weak(union, dir):
    recallset = []
    preset = []
    for i in range(len(union) - 1):
        data_union = list(union[0])
        data = not_delete(union[i + 1])
        tmp = []

        for val in tqdm(data):
            for val1 in data_union:
                s = mfun2(val, val1)
                if (s / len(val)) > 0.75:
                    tmp.append(val + '. ' + val1)
                    data_union.remove(val1)
                    break
        with open(dir + '/' + str(i) + '.txt', "w") as f:
            for line in tmp:
                f.write(line)
                f.write('\n')
        # accurcy = len(delete_repeat(tmp1)) / len(data_union)
        recall = len(tmp) / len(read_private_data())
        precision = len(tmp) / len(data)
        recallset.append(recall)
        preset.append(precision)

    return recallset, preset


def get_whole_accurcy_new(union, dir):
    recallset = []
    preset = []
    for i in range(len(union) - 1):
        data_union = list(union[0])
        data = not_delete(union[i + 1])
        tmp = []

        for val in tqdm(data):
            for val1 in data_union:
                s = find_lcsubstr(val, val1)
                if (len(s) / len(val)) > 0.75:
                    tmp.append(val + '. ' + val1)
                    data_union.remove(val1)
                    break
        with open(dir + '/' + str(i) + '.txt', "w") as f:
            for line in tmp:
                f.write(line)
                f.write('\n')
        # accurcy = len(delete_repeat(tmp1)) / len(data_union)
        recall = len(tmp) / len(read_private_data())
        precision = len(tmp) / len(data)
        recallset.append(recall)
        preset.append(precision)

    return recallset, preset


# def get_whole_accurcy(union, dir):
#     accurcyset = []
#     accset = []
#     data_union = delete_repeat(union[0])
#     for i in range(len(union) - 1):
#         data = not_delete(union[i + 1])
#         tmp = []
#         tmp1 = []
#         for val in tqdm(data):
#             for val1 in data_union:
#                 s = find_lcseque(val, val1)
#                 if (len(s) / len(val)) > 0.75 and (len(s) / len(val1)) > 0.75:
#                     tmp.append(val + '. ' + val1)
#                     tmp1.append(val)
#                     break
#         with open(dir + '/' + str(i) + '.txt', "w") as f:
#             for line in tmp:
#                 f.write(line)
#                 f.write('\n')
#         #accurcy = len(delete_repeat(tmp1)) / len(data_union)
#         accurcy = len(delete_repeat(tmp1)) / len(read_private_data())
#         acc = len(tmp) / len(data)
#         accurcyset.append(accurcy)
#         accset.append(acc)
#
#     return accurcyset, accset


def paint(Path, top_n, n_range, dir, private_data):


    if not os.path.exists(dir):
        os.makedirs(dir)

    key_union = extract_key_new(top_n, Path, n_range, private_data)

    for i in range(len(key_union) - 1):
        data = key_union[1:i+2]
        data = [token for st in data for token in st]
        key_union.append(data)

    accuracy, acc = get_whole_accurcy_new(key_union, dir)

    iter_list_recall = accuracy[:len(Path)]
    iter_list_precision = acc[:len(Path)]

    mix_list_recall = accuracy[len(Path):]
    mix_list_precision = acc[len(Path):]


    print('accuracy_iter:')
    head = '& '
    str = ''
    for i in range(len(iter_list_recall)):
        #print("{0:.4f}".format(iter_list_recall[i]))
        str = str + head + "{0:.4f}".format(iter_list_recall[i])
    print(str)
    str = ''
    print('========')
    for i in range(len(iter_list_precision)):
        #print("{0:.4f}".format(iter_list_precision[i]))
        str = str + head + "{0:.4f}".format(iter_list_precision[i])
    print(str)
    str = ''
    print('f-score')
    for i in range(len(iter_list_recall)):
        #print("{0:.4f}".format(fsocre(iter_list_recall[i], iter_list_precision[i])))
        str = str + head + "{0:.4f}".format(fsocre(iter_list_recall[i], iter_list_precision[i]))

    print(str)
    new_word = []
    str = ''
    print('accuracy_mix:')
    for i in range(len(mix_list_recall)):
        if i == 0:
            new_word.append(mix_list_recall[i])
        else:
            new_word.append(mix_list_recall[i] - mix_list_recall[i-1])
        #print("{0:.4f}".format(mix_list_recall[i]))
        str = str + head + "{0:.4f}".format(mix_list_recall[i])
    print(str)
    str = ''
    print('========')
    for i in range(len(mix_list_precision)):
        #print("{0:.4f}".format(mix_list_precision[i]))
        str = str + head + "{0:.4f}".format(mix_list_precision[i])
    print(str)
    str = ''
    print('f-score')
    for i in range(len(mix_list_precision)):
        #print("{0:.4f}".format(fsocre(mix_list_recall[i], mix_list_precision[i])))
        str = str + head + "{0:.4f}".format(fsocre(mix_list_recall[i], mix_list_precision[i]))

    print(str)
    print('new_words')
    str = ''
    for i in range(len(new_word)):
        str = str + head + "{0:.4f}".format(new_word[i])
    print(str)



save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path,
             iter_6_save_path, iter_7_save_path, iter_8_save_path, iter_9_save_path, iter_10_save_path]

path_new = '/medical_100000/k_20/block8'

Path_compare = [path + path_new + '/compare/zip_accept.txt' for path in save_path[1:]]
Path_our_method = [path + path_new + '/our_method/zip_accept.txt' for path in save_path]
Path_compare = Path_our_method[:1] + Path_compare

kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')

private_data = read_private_data()


top_1 = 1
top_2 = 2
n_range = (1, 1)

# Path_random = [Random_baseline + path_new + '/compare/all_public%d.txt' % i for i in range(10)]
#
#
# paint(Path_random, top_1, n_range, Random_baseline, private_data)

# print('zip_accept')
# print('top_1')
# print('compare')
# paint(Path_compare, top_1, n_range, keywords_save_path + path_new, private_data)


print('our_method')
paint(Path_our_method, top_1, n_range, keywords_save_path + path_new, private_data)


def read_data1(dirs):

    dataset = []
    for dir in dirs:
        with open(dir, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip('\n')
                dataset.append(line)
    return dataset

def collect(LM_path, path):
    file = ['XL_perplexity.txt', 'lower_case.txt', 'Zlib.txt']

    Path_LM_baseline = [LM_path + '/' + name for name in file]

    data = read_data1(Path_LM_baseline)

    lens = len(data) // 10

    if not os.path.exists(LM_path + path):
        os.makedirs(LM_path + path)

    for i in range(10):
        with open(LM_path + path + '/all_public%d.txt' % i, 'w+') as temp:
            for line in data[i * lens:(i + 1) * lens]:
                temp.write(line + '\n')
#
# collect(LM_baseline_movie, path_new)
#
# Path_LM_baseline = [LM_baseline_movie + path_new + '/all_public%d.txt' % i for i in range(10)]
#
# paint(Path_LM_baseline, top_1, n_range, LM_baseline_movie, private_data)

# path_new = '/medical_100000/k_20/limit_1'
# private_data = read_private_data()
#
# collect(LM_baseline_medical, path_new)
#
# Path_LM_baseline = [LM_baseline_medical + path_new + '/all_public%d.txt' % i for i in range(10)]
#
# paint(Path_LM_baseline, top_1, n_range, LM_baseline_medical, private_data)




