from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import csv
import re
from tqdm import tqdm
import pandas as pd
from collections import Counter
from utilize.dirpath import reconstruct_save_path_iter1,\
                            reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
                            reconstruct_save_path_iter4, reconstruct_save_path_iter5,\
                            reconstruct_save_path_iter6, reconstruct_save_path_iter7,\
                            reconstruct_save_path_iter8, reconstruct_save_path_iter9,\
                            reconstruct_save_path_iter10
from utilize.dirpath import target_save_path, iter_1_save_path, generic_save_path, \
                            iter_2_save_path, iter_3_save_path, iter_4_save_path, \
                            iter_5_save_path, iter_6_save_path, iter_7_save_path,\
                            iter_8_save_path, iter_9_save_path, iter_10_save_path


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


def read_data(dir):

    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            dataset.append(line)

    return dataset


def extract_information_no_prior(information, data):
    names = []
    organization = []
    location = []
    dates = []
    i = 0
    for list in information:
        name = ''
        organ = ''
        loc = ''
        date = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", data[i])
        if len(date) == 0:
            date = '#'
        else:
            date = date[0]
        for dic in list:
            if dic['entity'] == 'B-PER' or dic['entity'] == 'I-PER':
                if dic['word'].startswith('#'):
                    name = name.strip(' ') + dic['word'].strip('#')
                else:
                    name = name + ' ' + dic['word']
            if dic['entity'] == 'B-ORG' or dic['entity'] == 'I-ORG':
                if dic['word'].startswith('#'):
                    organ = organ.strip(' ') + dic['word'].strip('#')
                else:
                    organ = organ + ' ' + dic['word']
            if dic['entity'] == 'B-LOC' or dic['entity'] == 'I-LOC':
                if dic['word'].startswith('#'):
                    loc = loc.strip(' ') + dic['word'].strip('#')
                else:
                    loc = loc + ' ' + dic['word']
        if len(name) != 0:
            names.append(name.lstrip())
            if len(organ) != 0:
                organization.append(organ.lstrip())
            else:
                organization.append("#")
            if len(loc) != 0:
                location.append(loc.lstrip())
            else:
                location.append("#")
            if len(date) != 0:
                dates.append(date.lstrip())
            else:
                dates.append("#")

        i = i + 1
    return names, organization, location, dates



def extract_csv_no_prior(save_path, reconstruct_save_path):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("./bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)


    for i in range(len(reconstruct_save_path)):

        data = read_data(save_path[i] + '/zip_accept.txt')

        ner_results = nlp(data)

        names, organization, location, dates = extract_information_no_prior(ner_results, data)


        with open(reconstruct_save_path[i] + '/no_prior_reconstruct.csv', "w+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "country", "airline", "date"])
            row = []
            for j in range(len(names)):
                try:
                    tmp = []
                    tmp.append(names[j])
                    tmp.append(location[j])
                    tmp.append(organization[j])
                    tmp.append(dates[j])
                    row.append(tmp)
                except IndexError:
                    continue
            for r in row:
                writer.writerow(r)

def sky_eval_combine(reconstruct_save_path):

    for i in range(len(reconstruct_save_path)):
        airline_data_private = pd.read_csv("./data/private/skytrax/airline.csv")
        airline_data_reconstruct = pd.read_csv(reconstruct_save_path[i] + '/no_prior_reconstruct.csv')
        compare = []
        record = []
        with open(reconstruct_save_path[i] + '/no_prior.txt', 'w+') as f:
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

                        if idx == 100000:
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


path_new = '/sky_100000/limit_1_2/our_method'

reconstruct_save_path = [reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,
                         reconstruct_save_path_iter4, reconstruct_save_path_iter5, reconstruct_save_path_iter6,
                         reconstruct_save_path_iter7, reconstruct_save_path_iter8,
                         reconstruct_save_path_iter9, reconstruct_save_path_iter10]

reconstruct_save_path = [path + path_new for path in reconstruct_save_path]

save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path,
             iter_6_save_path, iter_7_save_path, iter_8_save_path, iter_9_save_path, iter_10_save_path]
save_path = [path + path_new for path in save_path]


extract_csv_no_prior(save_path, reconstruct_save_path)

sky_eval_combine(reconstruct_save_path)
