# -*-coding:utf-8-*-
import torch
from utilize.dirpath import sky_head, iter_1_save_path, generic_save_path, \
                            iter_2_save_path, iter_3_save_path, iter_4_save_path, \
                            iter_5_save_path, \
                            iter_1_model, iter_2_model, iter_3_model, iter_4_model, \
                            iter_5_model, iter_6_model, iter_7_model, iter_8_model,\
                            iter_9_model, iter_10_model, iter_6_save_path, iter_7_save_path,\
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
from sentence_transformers import SentenceTransformer
import os
from generate_accept_sentence_new import generate_main_diversity
from extract_iter_traindata import extract_accept, extract_zip_accept
from train_iter_model import train_and_save_part
from sentence_reconstruct import reconstruct_main
from bertNer import extract_name



path_new = '/sky_100000/sky_head/our_method'
save_path = [iter_1_save_path, iter_2_save_path, iter_3_save_path, iter_4_save_path, iter_5_save_path,
             iter_6_save_path, iter_7_save_path, iter_8_save_path, iter_9_save_path, iter_10_save_path]
save_path = [path + path_new for path in save_path]


reconstruct_save_path = [reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,
                         reconstruct_save_path_iter4, reconstruct_save_path_iter5,
                         reconstruct_save_path_iter6, reconstruct_save_path_iter7, reconstruct_save_path_iter8,
                         reconstruct_save_path_iter9, reconstruct_save_path_iter10]

reconstruct_save_path = [path + path_new for path in reconstruct_save_path]

name_save_path = [name_save_path_iter1, name_save_path_iter2, name_save_path_iter3, name_save_path_iter4,
                  name_save_path_iter5, name_save_path_iter6, name_save_path_iter7, name_save_path_iter8,
                  name_save_path_iter9, name_save_path_iter10]

name_save_path = [path + path_new for path in name_save_path]

sentence_model = SentenceTransformer('./all-MiniLM-L6-v2')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


BLOCK_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
NUMBER = 10000
limit = 0.5
top_k = 20
l = 2
sentence_num = 4
num = 100000

PATHS = [path + '/zip_accept.txt' for path in save_path]

Now_model = [generic_save_path, iter_1_model + path_new, iter_2_model + path_new,
             iter_3_model + path_new, iter_4_model + path_new, iter_5_model + path_new, iter_6_model + path_new,
             iter_7_model + path_new, iter_8_model + path_new, iter_9_model + path_new]


Next_model = [iter_1_model, iter_2_model, iter_3_model, iter_4_model, iter_5_model,
              iter_6_model, iter_7_model, iter_8_model, iter_9_model, iter_10_model]
Next_model = [model + path_new for model in Next_model]


for i in range(len(save_path)):

    if not os.path.exists(save_path[i]):
        os.makedirs(save_path[i])

    if not os.path.exists(reconstruct_save_path[i]):
        os.makedirs(reconstruct_save_path[i])

    if not os.path.exists(name_save_path[i]):
        os.makedirs(name_save_path[i])

    generate_main_diversity(sky_head, Now_model[i], device, save_path[i], NUMBER, top_k, l, sentence_num)
    extract_accept(save_path[i], limit, save_path)
    extract_zip_accept(save_path[i])
    extract_name(save_path[i], name_save_path[i])
    reconstruct_main(device, sky_head, name_save_path[i], reconstruct_save_path[i])
    train_and_save_part(Next_model[i], Now_model[i], PATHS[i:i+1], BLOCK_SIZE, BATCH_SIZE, EPOCHS, device, num, 'head')
    # generate_main_diversity(sky_block8, Next_model[i], device, save_path[i+1], NUMBER, top_k, l, sentence_num)
