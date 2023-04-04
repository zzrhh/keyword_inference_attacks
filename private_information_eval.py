# -*-coding:utf-8-*-
import pandas as pd
from tqdm import tqdm
from collections import Counter
from utilize.dirpath import reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
                            reconstruct_save_path_iter4, reconstruct_save_path_iter5,\
                            reconstruct_save_path_iter6, reconstruct_save_path_iter7,\
                            reconstruct_save_path_iter8, reconstruct_save_path_iter9,\
                            reconstruct_save_path_iter10
from extract_private_infomation import extract_sky_information_ner, extract_sky_information


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

def fsocre(recall, precision):

    if precision == 0 or recall == 0:
        return 0

    return 2 / (1 / precision + 1 / recall)

def sky_eval(reconstruct_save_path):

    for i in range(len(reconstruct_save_path)):
        airline_data_private = pd.read_csv("./data/private/skytrax/airline.csv")
        airline_data_reconstruct = pd.read_csv(reconstruct_save_path[i] + '/reconstruct.csv')
        compare = []
        record = []
        with open(reconstruct_save_path[i] + '/evalution.txt', 'w+') as f:
            for idx, index in tqdm(enumerate(airline_data_reconstruct['name'].index)):
                name = str(airline_data_reconstruct['name'].get(index))
                airline = str(airline_data_reconstruct['airline'].get(index))
                country = str(airline_data_reconstruct['country'].get(index))
                date = str(airline_data_reconstruct['date'].get(index))
                if name not in compare:
                    construct_dic = {}
                    for idx, index in enumerate(airline_data_private['airline_name'].index):

                        name_private = str(airline_data_private['author'].get(index))
                        airline_private = str(airline_data_private['airline_name'].get(index))
                        country_private = str(airline_data_private['author_country'].get(index))
                        date_private = str(airline_data_private['date'].get(index))
                        s_airline = find_lcseque(airline, airline_private)

                        if name == name_private:
                            construct_dic['name'] = name

                            if (len(s_airline) / len(airline)) > 0.75 and (len(s_airline) / len(airline_private)) > 0.75:
                                construct_dic['airline'] = airline

                            if country == country_private:
                                construct_dic['country'] = country

                            if date == date_private:
                                construct_dic['date'] = date

                        if idx == 10000:
                            break

                    if len(construct_dic) != 0:
                        information = ''
                        for key in construct_dic:
                            information = information + ',' + construct_dic[key]

                        f.write(information)
                        f.write('\n')
                        record.append(len(construct_dic))
                        compare.append(name)

            num_Count = Counter(record)

        print(num_Count)

def sky_eval_combine(reconstruct_save_path):

    for i in range(len(reconstruct_save_path)):
        airline_data_private = pd.read_csv("./data/private/skytrax/airline.csv")
        airline_data_reconstruct = pd.read_csv(reconstruct_save_path[i] + '/reconstruct_combine_ner.csv')
        compare = []
        record = []
        with open(reconstruct_save_path[i] + '/evalution_combine.txt', 'w+') as f:
            for idx, index in tqdm(enumerate(airline_data_reconstruct['name'].index)):
                name = str(airline_data_reconstruct['name'].get(index))
                airline = str(airline_data_reconstruct['airline'].get(index))
                country = str(airline_data_reconstruct['country'].get(index))
                date = str(airline_data_reconstruct['date'].get(index))
                if name not in compare:
                    construct_dic = {}
                    for idx, index in enumerate(airline_data_private['airline_name'].index):

                        name_private = str(airline_data_private['author'].get(index))
                        airline_private = str(airline_data_private['airline_name'].get(index))
                        country_private = str(airline_data_private['author_country'].get(index))
                        date_private = str(airline_data_private['date'].get(index))
                        s_airline_1 = find_lcseque(airline, airline_private)
                        s_airline_2 = find_lcseque(country, airline_private)

                        s_country_1 = find_lcseque(airline, country_private)
                        s_country_2 = find_lcseque(country, country_private)

                        if name == name_private and name != '#':
                            construct_dic['name'] = name

                            if len(s_airline_1) > len(s_airline_2):

                                if (len(s_airline_1) / len(airline)) > 0.75 and (len(s_airline_1) / len(airline_private)) > 0.75:
                                    construct_dic['airline'] = airline
                            else:
                                if (len(s_airline_2) / len(airline)) > 0.75 and (len(s_airline_2) / len(airline_private)) > 0.75:
                                    construct_dic['airline'] = country

                            if len(s_country_1) > len(s_country_2):
                                if (len(s_country_1) / len(country)) > 0.75 and (
                                        len(s_country_1) / len(country_private)) > 0.75:
                                    construct_dic['country'] = airline
                            else:
                                if (len(s_country_2) / len(country)) > 0.75 and (
                                        len(s_country_2) / len(country_private)) > 0.75:
                                    construct_dic['country'] = country

                            if date == date_private:
                                construct_dic['date'] = date

                        if idx == 10000:
                            break

                    if len(construct_dic) != 0:
                        information = ''
                        for key in construct_dic:
                            information = information + ',' + construct_dic[key]

                        f.write(information)
                        f.write('\n')
                        record.append(len(construct_dic))
                        compare.append(name)

            num_Count = Counter(record)

        print(num_Count)



# def movie_eval(reconstruct_save_path):
#
#
#     for i in range(len(reconstruct_save_path)):
#         airline_data_private = pd.read_csv("./data/private/Rotten tomatoes movie review/rotten_tomatoes_critic_reviews.csv")
#         airline_data_reconstruct = pd.read_csv(reconstruct_save_path[i] + '/reconstruct.csv')
#         compare = []
#         record = []
#         with open(reconstruct_save_path[i] + '//evalution.txt', 'w+') as f:
#             for idx, index in tqdm(enumerate(airline_data_reconstruct['critic_name'].index)):
#                 c_name = str(airline_data_reconstruct['critic_name'].get(index))
#                 p_name = str(airline_data_reconstruct['publisher_name'].get(index))
#                 r_date = str(airline_data_reconstruct['review_date'].get(index))
#
#                 if c_name not in compare:
#                     construct_dic = {}
#                     for idx, index in enumerate(airline_data_private['critic_name'].index):
#
#                         critic_name = str(airline_data_private['critic_name'].get(index))
#                         publisher_name = str(airline_data_private['publisher_name'].get(index))
#                         review_date = str(airline_data_private['review_date'].get(index))
#                         s_publish = find_lcseque(p_name, publisher_name)
#
#                         if c_name == critic_name and c_name != 'nan':
#                             construct_dic['name'] = c_name
#
#                             if (len(s_publish) / len(p_name)) > 0.75 and (len(s_publish) / len(publisher_name)) > 0.75:
#
#                                 construct_dic['publisher'] = p_name
#
#                             if r_date == review_date:
#
#                                 construct_dic['date'] = r_date
#
#                         if idx == 10000:
#                             break
#
#                     if len(construct_dic) != 0:
#                         information = ''
#                         for key in construct_dic:
#                             information = information + ',' + construct_dic[key]
#
#                         f.write(information)
#                         f.write('\n')
#                         record.append(len(construct_dic))
#                         compare.append(c_name)
#
#             num_Count = Counter(record)
#
#             print(num_Count)
#
# def movie_eval_combine(reconstruct_save_path):
#
#     for i in range(len(reconstruct_save_path)):
#         airline_data_private = pd.read_csv("./data/private/Rotten tomatoes movie review/rotten_tomatoes_critic_reviews.csv")
#         airline_data_reconstruct = pd.read_csv(reconstruct_save_path[i] + '/reconstruct_combine_ner.csv')
#         compare = []
#         record = []
#         with open(reconstruct_save_path[i] + '//evalution_combine.txt', 'w+') as f:
#             for idx, index in tqdm(enumerate(airline_data_reconstruct['critic_name'].index)):
#                 c_name = str(airline_data_reconstruct['critic_name'].get(index))
#                 p_name_or = str(airline_data_reconstruct['publisher_name_or'].get(index))
#                 p_name_lo = str(airline_data_reconstruct['publisher_name_lo'].get(index))
#                 r_date = str(airline_data_reconstruct['review_date'].get(index))
#
#                 if c_name not in compare:
#                     construct_dic = {}
#                     for idx, index in enumerate(airline_data_private['critic_name'].index):
#
#                         critic_name = str(airline_data_private['critic_name'].get(index))
#                         publisher_name = str(airline_data_private['publisher_name'].get(index))
#                         review_date = str(airline_data_private['review_date'].get(index))
#                         s_publish_or = find_lcseque(p_name_or, publisher_name)
#                         s_publish_lo = find_lcseque(p_name_lo, publisher_name)
#
#                         if c_name == critic_name and c_name != '#':
#                             construct_dic['name'] = c_name
#
#                             if len(s_publish_or) > len(s_publish_lo):
#                                 if (len(s_publish_or) / len(p_name_or)) > 0.75 and (
#                                         len(s_publish_or) / len(publisher_name)) > 0.75:
#
#                                     construct_dic['publisher'] = p_name_or
#                             else:
#                                 if (len(s_publish_lo) / len(p_name_lo)) > 0.75 and (
#                                         len(s_publish_lo) / len(publisher_name)) > 0.75:
#                                     construct_dic['publisher'] = p_name_lo
#
#                             if r_date == review_date:
#
#                                 construct_dic['date'] = r_date
#
#                         if idx == 10000:
#                             break
#
#                     if len(construct_dic) != 0:
#                         information = ''
#                         for key in construct_dic:
#                             information = information + ',' + construct_dic[key]
#
#                         f.write(information)
#                         f.write('\n')
#                         record.append(len(construct_dic))
#                         compare.append(c_name)
#
#             num_Count = Counter(record)
#
#             print(num_Count)


def eval(path):
    extract_sky_information_ner(path)
    extract_sky_information(path)
    sky_eval_combine(path)
    print('=================')
    sky_eval(path)


def private_eval():
    reconstruct_save_path = [reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,
                             reconstruct_save_path_iter4, reconstruct_save_path_iter5, reconstruct_save_path_iter6,
                             reconstruct_save_path_iter7, reconstruct_save_path_iter8,
                             reconstruct_save_path_iter9, reconstruct_save_path_iter10]

    path_new_our = '/sky_100000/block8/our_method'
    path_new_compare = '/sky_100000/block8/compare'

    reconstruct_save_path_compare = [path + path_new_compare for path in reconstruct_save_path[1:]]
    reconstruct_save_path_compare = [reconstruct_save_path_iter1 + path_new_our] + reconstruct_save_path_compare

    reconstruct_save_path_our = [path + path_new_our for path in reconstruct_save_path]

    eval(reconstruct_save_path_our)


private_eval()
