import numpy as np
import matplotlib.pyplot as plt

def compute_sum(list):

    list1 = list.copy()
    for i in range(len(list)):
        list[i] = sum(list1[:i+1])
    print(list)
    return list

def paint_mix_user(our, base, random, LM, savepath):

    # our = compute_sum(our)
    # base = compute_sum(base)
    # random = compute_sum(random)
    # LM = compute_sum(LM)


    name_list = [str(i + 1) for i in range(len(our))]

    x = list(range(len(our)))

    plt.figure(dpi=300, figsize=(10, 8))

    total_width, n = 0.8, 3
    width = total_width / n

    plt.bar(x, our, label='Ours', tick_label=name_list, color='brown', width=width)

    for i in range(len(x)):
        x[i] = x[i] + 0.1

    plt.bar(x, base, tick_label=name_list, label='Non-iteration', color='orange', width=width)

    for i in range(len(x)):
        x[i] = x[i] + 0.1

    plt.bar(x, random, tick_label=name_list, label='Random', color='blue', width=width)

    for i in range(len(x)):
        x[i] = x[i] + 0.1

    plt.bar(x, LM, tick_label=name_list, label='Sample recovery', color='black', width=width)


    plt.xticks(x, name_list, fontsize=22, fontweight='medium')
    plt.yticks(np.arange(0, 25, step=5), fontsize=22, fontweight='medium')
    plt.ylabel('Count', fontsize=26, fontweight='medium')
    plt.xlabel('Iteration', fontsize=26, fontweight='medium')

    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def fsocre(recall, precision):

    if precision == 0 or recall == 0:
        return 0

    return 2 / (1 / precision + 1 / recall)


def paint_mix_precision(our, base, random, LM, len_our, len_base, len_random, LM_len, total_num):

    our = compute_sum(our)
    base = compute_sum(base)
    random = compute_sum(random)
    len_our = compute_sum(len_our)
    len_base = compute_sum(len_base)
    len_random = compute_sum(len_random)

    precision_our = []
    precision_base = []
    precision_random = []

    recall_our = []
    recall_base = []
    recall_random = []

    f1_our = []
    f1_base = []
    f1_random = []

    for i in range(len(our)):
        precision_our.append(round(our[i] / len_our[i], 4))
        precision_base.append(round(base[i] / len_base[i], 4))
        precision_random.append(round(random[i] / len_random[i], 4))

        recall_our.append(round(our[i] / total_num, 4))
        recall_base.append(round(base[i] / total_num, 4))
        recall_random.append(round(random[i] / total_num, 4))

        f1_our.append(round(fsocre(recall_our[i], precision_our[i]), 4))
        f1_base.append(round(fsocre(recall_base[i], precision_base[i]), 4))
        f1_random.append(round(fsocre(recall_random[i], precision_random[i]), 4))

    print("precision_our:", precision_our)
    print("precision_base:", precision_base)
    print("precision_random:", precision_random)

    print('recall_our:', recall_our)
    print('recall_base:', recall_base)
    print('recall_random:', recall_random)

    print('f1_our:', f1_our)
    print('f1_base:', f1_base)
    print('f1_random:', f1_random)


def no_prior_compare(our, no_prior, savepath):
    our = compute_sum(our)
    no_prior = compute_sum(no_prior)


    name_list = [str(i + 1) for i in range(len(our))]

    x = list(range(len(our)))

    plt.figure(dpi=300, figsize=(10, 8))

    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, our, label='Ours', tick_label=name_list, color='brown', width=width)

    for i in range(len(x)):
        x[i] = x[i] + 0.1

    plt.bar(x, no_prior, tick_label=name_list, label='No contextual information', color='orange', width=width)


    plt.xticks(x, name_list, fontsize=22, fontweight='medium')
    plt.yticks(fontsize=22, fontweight='medium')
    plt.ylabel('Count', fontsize=26, fontweight='medium')
    plt.xlabel('Iteration', fontsize=26, fontweight='medium')

    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()

our_ner_rule = [3,5,8,13,8,6,16,16,20,12]
base_ner_rule = [3,3,2,3,2,4,2,2,6,1]
random_ner_rule = [4,0,2,1,2,5,3,2,2,0]
LM_ner_rule = [1,1,0,0,0,0,0,0,0,0]


our_rule_len = [59,93,148,181,240,229,268,304,259,240]
base_rule_len = [59,57,63,48,48,56,54,64,53,50]
random_rule_len = [49,49,55,52,50,56,49,48,51,15]
LM_rule_len = [47,52,56,49,47,61,47,48,56,53]

our_ner_len = [84,140,227,276,349,351,396,444,410,357]
base_ner_len = [84,87,82,85,72,87,96,86,83,86]
random_ner_len = [83,68,83,77,73,77,69,78,78,23]
LM_ner_len = [78,75,90,79,73,85,81,76,74,90]
our_ner_rule_len = np.sum([our_rule_len, our_ner_len], axis=0).tolist()
base_ner_rule_len = np.sum([base_rule_len, base_ner_len], axis=0).tolist()
random_ner_rule_len = np.sum([random_rule_len, random_ner_len], axis=0).tolist()
LM_ner_rule_len = np.sum([LM_rule_len, LM_ner_len], axis=0).tolist()

num = 40000

paint_mix_user(our_ner_rule, base_ner_rule, random_ner_rule, LM_ner_rule, './data/figs/sky_new_100000.pdf')

#paint_mix_precision(our_ner_rule, base_ner_rule, random_ner_rule, LM_ner_rule, our_ner_rule_len, base_ner_rule_len, random_ner_rule_len, LM_ner_rule_len, 40000)




our = [1,2,3,6,6,1,9,6,14,5]
no_prior = [0,0,0,0,0,0,1,1,0,1]

#no_prior_compare(our, no_prior, './data/figs/sky_no_prior.pdf')


sky_full = [3,5,8,13,8,6,16,16,20,12]
sky_head = [6,7,2,4,5,3,5,7,2,5]
sky_block4 = [4,4,14,11,15,8,13,15,21,15]
sky_block8 = [0,59,35,31,37,31,38,32,46,33]


#paint_mix_user(sky_full, sky_head, sky_block4, sky_block8, './data/figs/sky_finetuned_100000.pdf')
