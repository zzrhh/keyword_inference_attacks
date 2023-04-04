# -*-coding:utf-8-*-
import torch
from utilize.dirpath import target_save_path_sky_100000, iter_1_save_path, generic_save_path, \
                            iter_2_save_path, iter_3_save_path, iter_4_save_path, \
                            iter_5_save_path, iter_6_save_path, iter_7_save_path,\
                            iter_8_save_path, iter_9_save_path, iter_10_save_path
from utilize.dirpath import reconstruct_save_path_iter1,\
                            reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
                            reconstruct_save_path_iter4, reconstruct_save_path_iter5,\
                            reconstruct_save_path_iter6, reconstruct_save_path_iter7,\
                            reconstruct_save_path_iter8, reconstruct_save_path_iter9,\
                            reconstruct_save_path_iter10
from utilize.dirpath import name_save_path_iter1, name_save_path_iter2, name_save_path_iter3, \
                            name_save_path_iter4, name_save_path_iter5, name_save_path_iter6, \
                            name_save_path_iter7, name_save_path_iter8, name_save_path_iter9, \
                            name_save_path_iter10

import os
from generate_accept_sentence_new import generate_main_diversity
from extract_iter_traindata import extract_accept, extract_zip_accept
from sentence_reconstruct import reconstruct_main
from bertNer import extract_name


path_compare = '/sky_100000/limit_1_2/compare'
path_new = '/sky_100000/limit_1_2/our_method'

save_path1 = [iter_1_save_path + path_new, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path,
              iter_6_save_path, iter_7_save_path, iter_8_save_path, iter_9_save_path, iter_10_save_path]
save_path = [path + path_compare for path in save_path1[1:]]


reconstruct_save_path = [reconstruct_save_path_iter2, reconstruct_save_path_iter3,
                         reconstruct_save_path_iter4, reconstruct_save_path_iter5,reconstruct_save_path_iter6,
                         reconstruct_save_path_iter7, reconstruct_save_path_iter8,
                         reconstruct_save_path_iter9, reconstruct_save_path_iter10]
reconstruct_save_path = [path + path_compare for path in reconstruct_save_path]

name_save_path = [name_save_path_iter2, name_save_path_iter3, name_save_path_iter4,
                  name_save_path_iter5, name_save_path_iter6, name_save_path_iter7, name_save_path_iter8,
                  name_save_path_iter9, name_save_path_iter10]

name_save_path = [path + path_compare for path in name_save_path]


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BLOCK_SIZE = 256
NUMBER = 10000

Now_model = generic_save_path

sentence_num = 4
limit = 1
top_k = 20
l = 2


for i in range(len(save_path)):

    if not os.path.exists(save_path[i]):
        os.makedirs(save_path[i])

    if not os.path.exists(reconstruct_save_path[i]):
        os.makedirs(reconstruct_save_path[i])

    if not os.path.exists(name_save_path[i]):
        os.makedirs(name_save_path[i])

    generate_main_diversity(target_save_path_sky_100000, Now_model, device, save_path[i], NUMBER, top_k, l, sentence_num)
    extract_accept(save_path[i], limit, save_path1)
    extract_zip_accept(save_path[i])
    extract_name(save_path[i], name_save_path[i])
    reconstruct_main(device, target_save_path_sky_100000, name_save_path[i], reconstruct_save_path[i])

# path_compare = '/sky_100000/limit_1/compare'
# path_new = '/sky_100000/limit_1_2/our_method'
#
# path_copy = '/sky_100000/limit_1_2/compare'
#
# save_path1 = [iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path,
#               iter_6_save_path, iter_7_save_path, iter_8_save_path, iter_9_save_path, iter_10_save_path]
# save_path = [path + path_compare for path in save_path1]
#
# save_path2 = [path + path_copy for path in save_path1]
#
#
# reconstruct_save_path = [reconstruct_save_path_iter2, reconstruct_save_path_iter3,
#                          reconstruct_save_path_iter4, reconstruct_save_path_iter5,reconstruct_save_path_iter6,
#                          reconstruct_save_path_iter7, reconstruct_save_path_iter8,
#                          reconstruct_save_path_iter9, reconstruct_save_path_iter10]
# reconstruct_save_path = [path + path_copy for path in reconstruct_save_path]
#
# name_save_path = [name_save_path_iter2, name_save_path_iter3, name_save_path_iter4,
#                   name_save_path_iter5, name_save_path_iter6, name_save_path_iter7, name_save_path_iter8,
#                   name_save_path_iter9, name_save_path_iter10]
#
# name_save_path = [path + path_copy for path in name_save_path]
#
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# BLOCK_SIZE = 256
# NUMBER = 10000
#
# Now_model = generic_save_path
#
# sentence_num = 4
# limit = 0.5
# top_k = 20
# l = 2
# import shutil
# from glob import glob
#
# def mycopyfile(srcfile, dstpath):  # 复制函数
#     if not os.path.isfile(srcfile):
#         print("%s not exist!" % (srcfile))
#     else:
#         fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
#         if not os.path.exists(dstpath):
#             os.makedirs(dstpath)  # 创建路径
#         shutil.copy(srcfile, dstpath + fname)  # 复制文件
#         print("copy %s -> %s" % (srcfile, dstpath + fname))
#
# for i in range(len(save_path)):
#
#     if not os.path.exists(save_path2[i]):
#         os.makedirs(save_path2[i])
#
#     if not os.path.exists(reconstruct_save_path[i]):
#         os.makedirs(reconstruct_save_path[i])
#
#     if not os.path.exists(name_save_path[i]):
#         os.makedirs(name_save_path[i])
#
#     src_file_list = glob(save_path[i] + '/all.txt')  # glob获得路径下所有文件，可根据需要修改
#     for srcfile in src_file_list:
#         mycopyfile(srcfile, save_path1[i] + path_copy + '/')  # 复制文件
#
#     extract_accept(save_path2[i], limit, [iter_1_save_path + path_new])
#     extract_zip_accept(save_path2[i])
#     extract_name(save_path2[i], name_save_path[i])
#     reconstruct_main(device, target_save_path_sky_100000, name_save_path[i], reconstruct_save_path[i])
