# -*-coding:utf-8-*-
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utilize.dirpath import target_save_path, iter_1_save_path, generic_save_path
from tqdm import tqdm
from torch.autograd import Variable
from sentence_transformers import SentenceTransformer, util
import os
from torch import nn
import numpy as np
import scipy.stats
import math
import zlib


def gzip(data, truncate=True):

    e = float(float(len(zlib.compress(data.encode('utf-8'), 9))) / float(len(data)))

    e = float("{0:.4f}".format(e))
    if truncate and e > 1.0:
        e = 1.0
    return e


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


def compute_sensetive(P1, P2):
    p1 = sum(P1)
    p2 = sum(P2)
    sensetive = "{0:.4f}".format(abs(p1 - p2))
    return float(sensetive)

def compute_sensetive_KL(P1, P2):
    kL = scipy.stats.entropy(P1, P2)
    sensetive = "{0:.4f}".format(kL)
    return sensetive

def compute_sensetive_KL_sigmod(P1, P2):
    kL = scipy.stats.entropy(P1, P2)
    sensetive = "{0:.4f}".format(sigmoid(kL))
    return float(sensetive)

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def compute_sensetive_JS_divergence(p,q):
    p = normalization(np.array(p))
    q = normalization(np.array(q))
    M = (p+q)/2
    sensetive = 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
    sensetive = "{0:.4f}".format(sensetive)
    return sensetive


def select_top_k(predictions, k, t=0):

    prediction_k = predictions[0, -1, :].sort(descending=True)[1][:k+t]
    predicted_index = random.choice(list(enumerate(prediction_k)))[0]

    return prediction_k[predicted_index].item()


def find_text(predictions, index_text):
    softmax = nn.Softmax(dim=0)

    prediction_k = predictions[0, -1, :].sort(descending=True)[1]
    probablitys = predictions[0, -1, :].sort(descending=True)[0]

    probablitys = softmax(probablitys)

    a_t2n = prediction_k.cpu().numpy()

    index = np.argwhere(a_t2n == index_text)

    probability = probablitys[index[0][0]]

    return probability.item()


def generate_sentence_target(model, device, tokenizer, k=20, n=500):
    model.to(device)
    model.eval()

    text = '<|endoftext|>'

    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])

    total_predicted_text = ''

    for _ in range(n):
        tokens_tensor = Variable(tokens_tensor).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        predicted_index = select_top_k(predictions, k)

        total_predicted_text += tokenizer.decode(predicted_index)

        if '<|endoftext|>' in total_predicted_text:
            total_predicted_text = total_predicted_text.rstrip('<|endoftext|>')
            break

        if len(indexed_tokens) > 1023:
            break

        if '.' in total_predicted_text:
            break

        indexed_tokens += [predicted_index]
        tokens_tensor = torch.tensor([indexed_tokens])

    total_predicted_text = total_predicted_text.lstrip()

    return total_predicted_text


def generate_sentence_target_diversity(model, device, tokenizer, k=20, sentence_num=1):
    model.to(device)
    model.eval()

    text = '<|endoftext|>'

    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])

    total_predicted_text = ''
    n = 500

    for i in range(n):
        tokens_tensor = Variable(tokens_tensor).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        if i < 10:
            predicted_index = select_top_k(predictions, k, 20-i)
        else:
            predicted_index = select_top_k(predictions, k, 0)

        total_predicted_text += tokenizer.decode(predicted_index)

        if '<|endoftext|>' in total_predicted_text:
            total_predicted_text = total_predicted_text.rstrip('<|endoftext|>')
            break

        if len(indexed_tokens) > 128:
            break

        if total_predicted_text.count('.') == sentence_num:
            break

        indexed_tokens += [predicted_index]
        tokens_tensor = torch.tensor([indexed_tokens])

    total_predicted_text = total_predicted_text.lstrip()

    return total_predicted_text


def generate_sentence_probablity(model, device, tokenizer, generate_text, l):
    model.to(device)
    model.eval()

    total_predicted_text = ''

    probabilities = []

    str_list = generate_text.split()

    j = True
    for w in str_list[ : int(len(str_list) // l )]:
        if j:
            total_predicted_text = w
            j = False
        else:
            total_predicted_text = total_predicted_text + ' ' + w

    indexed_tokens = tokenizer.encode(total_predicted_text)

    tokens_tensor = torch.tensor([indexed_tokens])


    for w in str_list[int(len(str_list) // l ) :]:

        tokens_tensor = Variable(tokens_tensor).to(device)

        total_predicted_text = total_predicted_text + ' ' + w

        index = tokenizer.encode(total_predicted_text)


        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        probability = find_text(predictions, index[-1])
        probabilities.append(probability)

        # total_predicted_text += tokenizer.decode(index[-1])
        #
        # print(tokenizer.decode(index[-1]))
        #
        # print(total_predicted_text)

        indexed_tokens += [index[-1]]
        tokens_tensor = torch.tensor([indexed_tokens])

    return probabilities


def generate_main(target_save_path, generic_save_path, device, save_path, number, k, l):

    tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
    model_target = GPT2LMHeadModel.from_pretrained(target_save_path)

    tokenizer_generic = GPT2Tokenizer.from_pretrained(generic_save_path)
    model_generic = GPT2LMHeadModel.from_pretrained(generic_save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + '/all.txt', 'a+') as f:
        for i in tqdm(range(number)):
            text = generate_sentence_target(model_target, device, tokenizer_target, k)
            if len(text.split()) < 4:
                continue
            probability_generic = generate_sentence_probablity(model_generic, device, tokenizer_generic, text, l)
            probability_target = generate_sentence_probablity(model_target, device, tokenizer_generic, text, l)

            probability_target = -np.log(probability_target) + 1e-7
            probability_generic = -np.log(probability_generic) + 1e-7

            sens_res = compute_sensetive_KL(probability_generic, probability_target)
            f.write(text + ',' + sens_res + '\n')

def generate_main_diversity(target_save_path, generic_save_path, device, save_path, number, k, l, sentence_num):

    tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
    model_target = GPT2LMHeadModel.from_pretrained(target_save_path)

    tokenizer_generic = GPT2Tokenizer.from_pretrained(generic_save_path)
    model_generic = GPT2LMHeadModel.from_pretrained(generic_save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + '/all.txt', 'a+') as f:
        for i in tqdm(range(number)):
            text = generate_sentence_target_diversity(model_target, device, tokenizer_target, k, sentence_num)
            if len(text.split()) < 4:
                continue
            probability_generic = generate_sentence_probablity(model_generic, device, tokenizer_generic, text, l)
            probability_target = generate_sentence_probablity(model_target, device, tokenizer_generic, text, l)

            probability_target = -np.log(probability_target) + 1e-7
            probability_generic = -np.log(probability_generic) + 1e-7
            sens_res = '0.0000'
            sens_res = compute_sensetive_KL(probability_generic, probability_target)
            f.write(text + ',' + sens_res + '\n')


def generate_baseline_data(target_save_path, device, k, sentence_num, number, path):
    tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
    model_target = GPT2LMHeadModel.from_pretrained(target_save_path)

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/all.txt', 'w+') as f:
        for i in tqdm(range(number)):
            text = generate_sentence_target_diversity(model_target, device, tokenizer_target, k, sentence_num)
            if len(text.split()) < 4:
                continue
            f.write(text + '\n')

# def generate_main_diversity_js(target_save_path, generic_save_path, device, save_path, number):
#
#     tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
#     model_target = GPT2LMHeadModel.from_pretrained(target_save_path)
#
#     tokenizer_generic = GPT2Tokenizer.from_pretrained(generic_save_path)
#     model_generic = GPT2LMHeadModel.from_pretrained(generic_save_path)
#
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     with open(save_path + '/all.txt', 'a+') as f:
#         for i in tqdm(range(number)):
#             text = generate_sentence_target_diversity(model_target, device, tokenizer_target)
#             if len(text.split()) < 4:
#                 continue
#             probability_generic = generate_sentence_probablity(model_generic, device, tokenizer_generic, text)
#             probability_target = generate_sentence_probablity(model_target, device, tokenizer_generic, text)
#
#             probability_target = -np.log(probability_target)
#             probability_generic = -np.log(probability_generic)
#
#             sens_res = compute_sensetive_JS_divergence(probability_generic, probability_target)
#             f.write(text + ',' + sens_res + '\n')
#
#
def text_KL(target_save_path, generic_save_path, device, text, l):

    tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
    model_target = GPT2LMHeadModel.from_pretrained(target_save_path)

    tokenizer_generic = GPT2Tokenizer.from_pretrained(generic_save_path)
    model_generic = GPT2LMHeadModel.from_pretrained(generic_save_path)


    probability_generic = generate_sentence_probablity(model_generic, device, tokenizer_generic, text, l)
    probability_target = generate_sentence_probablity(model_target, device, tokenizer_target, text, l)

    probability_target = -np.log(probability_target) + 1e-7
    probability_generic = -np.log(probability_generic) + 1e-7

    sens_res = compute_sensetive_KL(probability_generic, probability_target)

    print(sens_res)

def read_private_data():
    private_data = []
    with open('./data/private/train.txt', 'r') as f:

        for idx, line in enumerate(f):
            li = line.strip()
            lin = str(li).lstrip("BACKGROUND OBJECTIVE METHODS RESULTS CONCLUSIONS")
            lin = lin.strip()
            if lin != '' and lin[0:3] != '###':

                private_data.append(lin.strip('\n'))

            if idx == 10000:
                break

    return private_data

def generate_trainingdata_sensetive(target_save_path, generic_save_path, device, save_path, l):
    privatedata = read_private_data()
    tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
    model_target = GPT2LMHeadModel.from_pretrained(target_save_path)

    tokenizer_generic = GPT2Tokenizer.from_pretrained(generic_save_path)
    model_generic = GPT2LMHeadModel.from_pretrained(generic_save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + '/train_sensetive.txt', 'a+') as f:
        for i in tqdm(range(len(privatedata))):
            text = privatedata[i]
            if len(text.split()) < 4:
                continue
            probability_generic = generate_sentence_probablity(model_generic, device, tokenizer_generic, text, l)
            probability_target = generate_sentence_probablity(model_target, device, tokenizer_target, text, l)

            probability_target = -np.log(probability_target) + 1e-7
            probability_generic = -np.log(probability_generic) + 1e-7

            sens_res = compute_sensetive_KL(probability_generic, probability_target)
            f.write(text + ',' + sens_res + '\n')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# text = 'The peripheral blood progenitor cell ( PBPC ) mobilization capacity of EPO in association with either G-CSF or sequential GM-CSF/G-CSF was compared in a randomized fashion after epirubicin , paclitaxel , and cisplatin ( ETP ) chemotherapy .'
# text1 = 'Single procedure success rates of pulmonary vein isolation ( PVI ) in patients with paroxysmal atrial fibrillation ( PAF ) are still unsatisfactory .'
# l = 2
# #savepath = './data/private'
# text_KL(target_save_path, generic_save_path, device, text1, l)
# # generate_trainingdata_sensetive(target_save_path, generic_save_path, device, savepath, l)
