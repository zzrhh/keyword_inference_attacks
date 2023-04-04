# -*-coding:utf-8-*-
from utilize.dirpath import reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
                            reconstruct_save_path_iter4, reconstruct_save_path_iter5,\
                            reconstruct_save_path_iter6, reconstruct_save_path_iter7,\
                            reconstruct_save_path_iter8, reconstruct_save_path_iter9,\
                            reconstruct_save_path_iter10, LM_baseline_sky
import pandas as pd


new_words = []
repeat_sentence = []

def build_new_words(filea):
    fa = open(filea)
    a = fa.readlines()
    fa.close()

    a = [i.lstrip(",") for i in a]
    new_words.extend([i.split(',')[0] for i in a])


def eval_new(fileb):

    fb = open(fileb)
    b = fb.readlines()
    fb.close()

    fa = open(fileb)
    a = fa.readlines()
    fa.close()

    b = [i.lstrip(",") for i in b]
    b = [i.split(',')[0] for i in b]

    d = [i for i in a]
    c = [i.strip('\n') for i in b if i not in new_words]

    repeat = [i.strip('\n') for i in b if i in new_words]

    repeat_sentence.extend(repeat)

    new_words.extend(c)

    return len(c)

def eval_count(file):
    fb = open(file)
    b = fb.readlines()
    fb.close()

    b = [i.count(",") for i in b]

    dic = {"1":0, "2":0, "3":0}

    for x in b:
        if x == 1:
            dic['1'] += 1
        if x == 2:
            dic['2'] += 1
        if x == 3:
            dic['3'] += 1

    print(dic)


def eval_precision(reconstruct_save_path, fileb, key):

    reconstruct = pd.read_csv(reconstruct_save_path)

    fb = open(fileb)
    b = fb.readlines()
    fb.close()

    b = [i.lstrip(",") for i in b]
    b = [i.split(',')[0].strip('\n') for i in b]

    count = 0

    name = []

    for i in range(len(reconstruct)):
        name.append(str(reconstruct[key][i]))

    for i in range(len(b)):
        if b[i] in name:
            count +=1

    precision = count / len(reconstruct)

    return precision, len(reconstruct)



reconstruct_save_path = [reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,
                         reconstruct_save_path_iter4, reconstruct_save_path_iter5, reconstruct_save_path_iter6,
                         reconstruct_save_path_iter7, reconstruct_save_path_iter8,
                         reconstruct_save_path_iter9, reconstruct_save_path_iter10]

path_new_our = '/sky_100000/limit_1_2/our_method'
path_new_compare = '/sky_100000/limit_1_2/compare'
path_new_random = '/Random/sky_100000/limit_1_2/compare'
path_new_LM = '/limit_1_2/compare'

new = path_new_our

reconstruct_save_path_rule = [path + new + '/evalution.txt' for path in reconstruct_save_path]

reconstruct_save_path_ner = [path + new + '/evalution_combine.txt' for path in reconstruct_save_path]

reconstruct_save_path_rule_compare = [path + path_new_compare + '/evalution.txt' for path in reconstruct_save_path[1:]]

reconstruct_save_path_ner_compare = [path + path_new_compare + '/evalution_combine.txt' for path in reconstruct_save_path[1:]]

reconstruct_save_path_rule_compare = [reconstruct_save_path_iter1 + path_new_our + '/evalution.txt'] + reconstruct_save_path_rule_compare

reconstruct_save_path_ner_compare = [reconstruct_save_path_iter1 + path_new_our + '/evalution_combine.txt'] + reconstruct_save_path_ner_compare

reconstruct_rule = [path + new + '/reconstruct.csv' for path in reconstruct_save_path]

reconstruct_ner = [path + new + '/reconstruct_combine_ner.csv' for path in reconstruct_save_path]

reconstruct_rule_compare = [path + path_new_compare + '/reconstruct.csv' for path in reconstruct_save_path[1:]]

reconstruct_ner_compare = [path + path_new_compare + '/reconstruct_combine_ner.csv' for path in reconstruct_save_path[1:]]

reconstruct_rule_compare = [reconstruct_save_path_iter1 + path_new_our + '/reconstruct.csv'] + reconstruct_rule_compare

reconstruct_ner_compare = [reconstruct_save_path_iter1 + path_new_our + '/reconstruct_combine_ner.csv'] + reconstruct_ner_compare

reconstruct_save_path_no_prior = [path + path_new_our + '/no_prior.txt' for path in reconstruct_save_path]


s = ''

for i in range(len(reconstruct_save_path_rule)):
    a = eval_new(reconstruct_save_path_rule[i])
    b = eval_new(reconstruct_save_path_ner[i])
    eval_count(reconstruct_save_path_no_prior[i])
    print("rule + ner:", a+b)
    s = s + str(a+b) + ','


# for i in range(len(reconstruct_save_path_rule)):
#     a,b = eval_precision(reconstruct_rule[i], reconstruct_save_path_rule[i], 'name')
#     print(b)
#     s = s + str(b) + ','

print(s)
