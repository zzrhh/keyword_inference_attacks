# -*-coding:utf-8-*-
import torch
from utilize.dirpath import medical_head, iter_1_save_path, generic_save_path, \
                            iter_2_save_path, iter_3_save_path, iter_4_save_path, \
                            iter_5_save_path, iter_6_save_path, iter_7_save_path,\
                            iter_8_save_path, iter_9_save_path, iter_10_save_path,\
                            iter_1_model, iter_2_model, iter_3_model, iter_4_model, \
                            iter_5_model, iter_6_model, iter_7_model, iter_8_model,\
                            iter_9_model, iter_10_model


import os
from generate_accept_sentence_new import generate_main_diversity
from extract_iter_traindata import extract_accept, extract_zip_accept
from train_iter_model import train_and_save, train_and_save_part


path_new = '/medical_100000/k_20/head/our_method'
save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path,
             iter_6_save_path, iter_7_save_path, iter_8_save_path, iter_9_save_path, iter_10_save_path]
save_path = [path + path_new for path in save_path]


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
BLOCK_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
NUMBER = 20000
PATHS = [path + '/zip_accept.txt' for path in save_path]

Now_model = [generic_save_path, iter_1_model + path_new, iter_2_model + path_new,
             iter_3_model + path_new, iter_4_model + path_new, iter_5_model + path_new, iter_6_model + path_new,
             iter_7_model + path_new, iter_8_model + path_new, iter_9_model + path_new]


Next_model = [iter_1_model, iter_2_model, iter_3_model, iter_4_model, iter_5_model,
              iter_6_model, iter_7_model, iter_8_model, iter_9_model, iter_10_model]
Next_model = [model + path_new for model in Next_model]


limit = 1
top_k = 20
l = 2
sentence_num = 1
num = 100000

part = 'head'


for i in range(len(save_path)):

    if not os.path.exists(save_path[i]):
        os.makedirs(save_path[i])

    #generate_main_diversity(medical_head, Now_model[i], device, save_path[i], NUMBER, top_k, l, sentence_num)
    extract_accept(save_path[i], limit, save_path)
    extract_zip_accept(save_path[i])
    train_and_save_part(Next_model[i], Now_model[i], PATHS[i:i + 1], BLOCK_SIZE, BATCH_SIZE, EPOCHS, device, num, part)
    generate_main_diversity(medical_head, Next_model[i], device, save_path[i+1], NUMBER, top_k, l, sentence_num)
