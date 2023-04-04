# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

#medical
mix_compare_r = [0.0021, 0.0044, 0.0066, 0.0087, 0.0111, 0.0134, 0.0156, 0.0179, 0.0200, 0.0225]
mix_compare_p = [0.6808, 0.6882, 0.6976, 0.6957, 0.6995, 0.7009, 0.6980, 0.6994, 0.6952, 0.7007]
mix_compare_f = [0.0041, 0.0087, 0.0131, 0.0172, 0.0218, 0.0263, 0.0305, 0.0349, 0.0389, 0.0436]
mix_compare_new_r = [0.0021, 0.0023, 0.0022, 0.0021, 0.0024, 0.0023, 0.0022, 0.0023, 0.0021, 0.0025]


mix_our_r = [0.0021, 0.0049, 0.0087, 0.0126, 0.0171, 0.0213, 0.0254, 0.0301, 0.0344, 0.0391]
mix_our_p = [0.6808, 0.7155, 0.7154, 0.7156, 0.7205, 0.7190, 0.7220, 0.7179, 0.7172, 0.7190]
mix_our_f = [0.0041, 0.0097, 0.0171, 0.0248, 0.0334, 0.0414, 0.0491, 0.0578, 0.0657, 0.0742]
mix_our_new_r = [0.0021, 0.0028, 0.0038, 0.0040, 0.0045, 0.0042, 0.0041, 0.0047, 0.0043, 0.0047]


Random_r = [0.0021, 0.0042, 0.0061, 0.0081, 0.0102, 0.0123, 0.0146, 0.0168, 0.0188, 0.0210]
Random_p = [0.6808, 0.6808, 0.6641, 0.6673, 0.6700, 0.6724, 0.6846, 0.6899, 0.6863, 0.6892]
Random_f = [0.0041, 0.0083, 0.0120, 0.0161, 0.0201, 0.0242, 0.0286, 0.0328, 0.0367, 0.0408]
Random_new_r = [0.0021, 0.0021, 0.0019, 0.0021, 0.0021, 0.0021, 0.0023, 0.0022, 0.0020, 0.0022]

LM_r = [0.0023, 0.0047, 0.0069, 0.0092, 0.0115, 0.0137, 0.0160, 0.0181, 0.0205, 0.0226]
LM_p = [0.7106, 0.7271, 0.7167, 0.7170, 0.7179, 0.7125, 0.7122, 0.7083, 0.7114, 0.7051]
LM_f = [0.0045, 0.0092, 0.0136, 0.0181, 0.0226, 0.0268, 0.0312, 0.0354, 0.0398, 0.0437]
LM_new_r = [0.0023, 0.0024, 0.0022, 0.0023, 0.0023, 0.0022, 0.0023, 0.0022, 0.0024, 0.0021]

#movie
mix_compare_r_movie = []
mix_compare_p_movie = []
mix_compare_f_movie = []
mix_compare_new_r_movie = []


mix_our_r_movie = []
mix_our_p_movie = []
mix_our_f_movie = []
mix_our_new_r_movie = []


Random_r_movie = []
Random_p_movie = []
Random_f_movie = []
Random_new_r_movie = []


LM_r_movie = []
LM_p_movie = []
LM_f_movie = []
LM_new_r_movie = []


def paint(our_method, base_line, random, LM, ylabel, savepath):

    name_list = [str(i+1) for i in range(len(our_method))]

    x = list(range(len(our_method)))

    plt.figure(dpi=300, figsize=(10, 8))

    plt.plot(x, our_method, label='Ours', color='blue', marker='o', linestyle='-', ms=12, linewidth=3)
    plt.plot(x, base_line, label='Non-iteration', color='orange', marker='s', linestyle='--', ms=12, linewidth=3)
    plt.plot(x, random, label='Random', color='brown', marker='*', linestyle='--', ms=12, linewidth=3)
    plt.plot(x, LM, label='Sample recovery', color='black', marker='.', linestyle='--', ms=12, linewidth=3)


    plt.xticks(x, name_list, fontsize=22, fontweight='medium')
    plt.yticks(fontsize=22, fontweight='medium')
    plt.ylabel(ylabel, fontsize=26, fontweight='medium')
    plt.xlabel('Iteration', fontsize=26, fontweight='medium')
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()


recall = 'Attack Recall'
precision = 'Attack Precision'
f1 = 'Attack F1 score'


paint(mix_our_r, mix_compare_r, Random_r, LM_r, recall, './data/figs/sens_recall_100000.pdf')
paint(mix_our_p, mix_compare_p, Random_p, LM_p, precision, './data/figs/sens_precision_100000.pdf')
paint(mix_our_f, mix_compare_f, Random_f, LM_f, f1, './data/figs/sens_f1_100000.pdf')
paint(mix_our_new_r, mix_compare_new_r, Random_new_r, LM_new_r, recall, './data/figs/sens_new_100000.pdf')

# paint_new(mix_our_r_movie, mix_compare_r_movie, Random_r_movie, recall, './data/figs/sens_recall_movie.pdf')
# paint_new(mix_our_p_movie, mix_compare_p_movie, Random_p_movie, precision, './data/figs/sens_precision_movie.pdf')
# paint_new(mix_our_f_movie, mix_compare_f_movie, Random_f_movie, f1, './data/figs/sens_f1_movie.pdf')
# paint_new(mix_our_new_r_movie, mix_compare_new_r_movie, Random_new_r_movie, recall, './data/figs/sens_new_movie.pdf')

l_1_2 = [0.0041, 0.0097, 0.0171, 0.0248, 0.0334, 0.0414, 0.0491, 0.0578, 0.0657, 0.0742]
l_2_3 = [0.0042, 0.0154, 0.0275, 0.0395, 0.0528, 0.0659, 0.0791, 0.0911, 0.1049, 0.1175]
l_1_3 = [0.0041, 0.0070, 0.0119, 0.0171, 0.0222, 0.0274, 0.0331, 0.0388, 0.0445, 0.0504]


l_1_2p = [0.6808, 0.7155, 0.7154, 0.7156, 0.7205, 0.7190, 0.7220, 0.7179, 0.7172, 0.7190]
l_2_3p = [0.6808, 0.7289, 0.7194, 0.7280, 0.7245, 0.7247, 0.7264, 0.7231, 0.7234, 0.7221]
l_1_3p = [0.6808, 0.7078, 0.7197, 0.7269, 0.7400, 0.7343, 0.7314, 0.7341, 0.7328, 0.7307]

l_1_2r = [0.0021, 0.0049, 0.0087, 0.0126, 0.0171, 0.0213, 0.0254, 0.0301, 0.0344, 0.0391]
l_2_3r = [0.0021, 0.0078, 0.0140, 0.0203, 0.0274, 0.0344, 0.0416, 0.0484, 0.0563, 0.0636]
l_1_3r = [0.0021, 0.0035, 0.0060, 0.0087, 0.0113, 0.0140, 0.0169, 0.0199, 0.0230, 0.0261]


delta_1 = [0.0097, 0.0282, 0.0461, 0.0653, 0.0836, 0.1019, 0.121, 0.139, 0.1565, 0.174]
delta_2 = [0.0041, 0.0097, 0.0171, 0.0248, 0.0334, 0.0414, 0.0491, 0.0578, 0.0657, 0.0742]
delta_3 = [0.0020, 0.0056, 0.0098, 0.0141, 0.0185, 0.0232, 0.0277, 0.0328, 0.0377, 0.0422]

delta_1p = [0.6785, 0.7077, 0.7087, 0.7209, 0.7167, 0.7215, 0.7211, 0.7249, 0.7233, 0.7238]
delta_2p = [0.6808, 0.7155, 0.7154, 0.7156, 0.7205, 0.7190, 0.7220, 0.7179, 0.7172, 0.7190]
delta_3p = [0.7073, 0.6808, 0.6846, 0.6984, 0.7102, 0.7092, 0.7076, 0.7122, 0.7139, 0.7154]

delta_1r = [0.0049, 0.0144, 0.0237, 0.0341, 0.0443, 0.0547, 0.0658, 0.0764, 0.0872, 0.0981]
delta_2r = [0.0021, 0.0049, 0.0087, 0.0126, 0.0171, 0.0213, 0.0254, 0.0301, 0.0344, 0.0391]
delta_3r = [0.0010, 0.0028, 0.0049, 0.0071, 0.0094, 0.0118, 0.0141, 0.0168, 0.0194, 0.0217]



name_list = list(range(1, len(l_1_2) + 1))

x = list(range(len(l_1_2)))

plt.figure(dpi=300, figsize=(10, 8))

# plt.plot(x, l_1_2p, label='l=1/2', color='orange', marker='o', linestyle='--', ms=12, linewidth=3)
# plt.plot(x, l_1_3p, label='l=1/3', color='blue', marker='s', linestyle='--', ms=12, linewidth=3)
# plt.plot(x, l_2_3p, label='l=2/3', color='brown', marker='p', linestyle='--', ms=12, linewidth=3)


# plt.plot(x, delta_1p, label='δ=0.5', color='orange', marker='o', linestyle='--', ms=12, linewidth=3)
# plt.plot(x, delta_2p, label='δ=1', color='blue', marker='s', linestyle='--', ms=12, linewidth=3)
# plt.plot(x, delta_3p, label='δ=1.5', color='brown', marker='p', linestyle='--', ms=12, linewidth=3)
#
#
#
# plt.xticks(x, name_list, fontsize=22, fontweight='medium')
# plt.yticks(fontsize=22, fontweight='medium')
# plt.ylabel(precision, fontsize=26, fontweight='medium')
# plt.xlabel('Iteration', fontsize=26, fontweight='medium')
# plt.legend(fontsize=22)
# plt.tight_layout()
# plt.savefig('./data/figs/delta_p_100000.pdf', bbox_inches='tight')
# plt.show()

