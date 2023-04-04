# -*-coding:utf-8-*-
from tqdm import tqdm
import pandas as pd
import csv

PATH = './data/private/Rotten tomatoes movie review/rotten_tomatoes_critic_reviews.csv'

new_path = './data/private/Rotten tomatoes movie review/movie_critic_reviews.txt'

with open(PATH, 'r') as file:
    reader = csv.DictReader(file)
    column = [row['review_content'] for row in reader]

i = 0
with open(new_path, 'w+') as f:
    for toke in column:
        i = i + 1
        f.write(toke + '\n')

