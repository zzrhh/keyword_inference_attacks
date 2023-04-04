# -*-coding:utf-8-*-
import torch
from utilize.dirpath import target_save_path_movie_review_100000, iter_1_save_path, generic_save_path, \
                            iter_2_save_path, iter_3_save_path, iter_4_save_path, \
                            iter_5_save_path, iter_6_save_path, iter_7_save_path,\
                            iter_8_save_path, iter_9_save_path, iter_10_save_path

import os
from generate_accept_sentence_new import generate_main_diversity
from extract_iter_traindata import extract_accept, extract_zip_accept


path_compare = '/movie_review_100000/k_20/limit_1/compare'
path_new = '/movie_review_100000/k_20/limit_1/our_method'
save_path_generic = [iter_1_save_path + path_new, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path,
                     iter_6_save_path, iter_7_save_path, iter_8_save_path, iter_9_save_path, iter_10_save_path]
save_path = [path + path_compare for path in save_path_generic[1:]]


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

NUMBER = 20000
Now_model = generic_save_path

limit = 1
top_k = 20
l = 2
sentence_num = 1

for i in range(len(save_path)):

    i = i + 5
    if not os.path.exists(save_path[i]):
        os.makedirs(save_path[i])

    generate_main_diversity(target_save_path_movie_review_100000, Now_model, device, save_path[i], NUMBER, top_k, l, sentence_num)
    extract_accept(save_path[i], limit, save_path_generic)
    extract_zip_accept(save_path[i])



# import time
#
# while True:
#     time_now = time.strftime("%H:%M:%S", time.localtime())  # 刷新
#     if time_now == "05:10:50":  # 此处设置每天定时的时间
#
#         # 此处3行替换为需要执行的动作
#         for i in range(len(save_path)):
#             if not os.path.exists(save_path[i]):
#                 os.makedirs(save_path[i])
#
#             generate_main_diversity(target_save_path, Now_model, device, save_path[i], NUMBER, top_k, l)
#             extract_accept(save_path[i], limit, save_path)
#             extract_zip_accept(save_path[i])
#         time.sleep(2)  # 因为以秒定时，所以暂停2秒，使之不会在1秒内执行多次

