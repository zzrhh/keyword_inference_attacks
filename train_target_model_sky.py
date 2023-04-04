# -*-coding:utf-8-*-
from train_iter_model import train_and_save_part
from utilize.dirpath import sky_block8, generic_save_path
import torch


BLOCK_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
num = 100000
#PATHS = ["./data/private/Rotten tomatoes movie review/movie_critic_reviews.txt"]
PATHS = ["./data/private/skytrax/airline.txt"]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


train_and_save_part(sky_block8, generic_save_path, PATHS, BLOCK_SIZE, BATCH_SIZE, EPOCHS, device, num, 'block8')

