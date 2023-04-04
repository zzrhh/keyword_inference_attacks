# -*-coding:utf-8-*-
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import time
from utilize.dirpath import generic_save_path, iter_1_model, iter_1_save_path
import os
from tqdm import tqdm

'''
BLOCK_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
PATHS = [iter_1_save_path + '/accept.txt']
Now_model = generic_save_path
Next_model = iter_1_model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if not os.path.exists(Next_model):
    os.makedirs(Next_model)
'''
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

def read_data(dir,tokenizer,num):

    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            lin = str(line).rstrip("<|endoftext|>")
            lin = lin + tokenizer.eos_token
            dataset.append(lin)

            if idx == num:
                print('sentence number:', num)
                break
    return dataset

def encode_string(paths, tokenizer, num):

    string_tokenized = []
    for filename in paths:
        doc = read_data(filename, tokenizer, num)
        for text in doc:
            encode_idx = tokenizer.encode(text)
            string_tokenized.extend(encode_idx)

    return string_tokenized

def enumerate_count(file_name):
    with open(file_name) as f:
        for count, _ in enumerate(f, 1):
            pass
    return count

def encode_string_privacy(paths, tokenizer, accept):

    string_tokenized = []
    i = 0
    max_ = [0]
    maxmax = 0
    for filename in paths:
        count = 0
        for j in range(len(accept)):
            max = enumerate_count(accept[j])
            maxmax = maxmax + max
            max_.append(maxmax)
        i = i + 1
        doc = read_private_data()
        for text in doc[max_[-2]:max_[-1]]:
            count = count + 1
            encode_idx = tokenizer.encode(text)
            string_tokenized.extend(encode_idx)
            # if count > max_:
            #     break
    print(len(string_tokenized))
    return string_tokenized


def split_input_labels(string_tokenized, block_size):
    examples = []
    for i in range(0, len(string_tokenized) - block_size + 1, block_size):
        examples.append(string_tokenized[i:i + block_size])

    inputs = torch.tensor(examples)
    labels = torch.tensor(examples)

    print(inputs.shape, labels.shape)
    return inputs, labels


def train(model, train_loader, epochs, optimizer, device):

    for epoch in range(epochs):

        loop = tqdm(train_loader, leave=True)

        for (data, target) in loop:
            # for batch in loop:
            data, target = Variable(data).to(device), Variable(target).to(device)

            optimizer.zero_grad()

            outputs = model(data, labels=target)

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


def train_and_save(Next_model, Now_model, PATHS, BLOCK_SIZE, BATCH_SIZE, EPOCHS, device, num):

    if not os.path.exists(Next_model):
        os.makedirs(Next_model)

    tokenizer = GPT2Tokenizer.from_pretrained(Now_model)
    model = GPT2LMHeadModel.from_pretrained(Now_model)

    dataset = encode_string(PATHS, tokenizer, num)

    inputs, labels = split_input_labels(dataset, BLOCK_SIZE)

    train_set = TensorDataset(inputs, labels)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    print(device.type)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器

    train(model, train_loader, EPOCHS, optimizer, device)

    model.save_pretrained(Next_model)
    tokenizer.save_pretrained(Next_model)


def train_and_save_part(Next_model, Now_model, PATHS, BLOCK_SIZE, BATCH_SIZE, EPOCHS, device, num, part):

    if not os.path.exists(Next_model):
        os.makedirs(Next_model)

    tokenizer = GPT2Tokenizer.from_pretrained(Now_model)
    model = GPT2LMHeadModel.from_pretrained(Now_model)

    dataset = encode_string(PATHS, tokenizer, num)

    inputs, labels = split_input_labels(dataset, BLOCK_SIZE)

    train_set = TensorDataset(inputs, labels)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    print(device.type)

    model.to(device)
    model.train()

    # for name, param in model.named_parameters():
    #     print(name, '      ', param.size())

    head = [2, 13]
    block4 = [98, 145]
    block8 = [50, 145]

    print(part)

    if part == 'head':
        block = head
    elif part == 'block4':
        block = block4
    else:
        block = block8

    print(block)

    for i, param in enumerate(model.parameters()):
        param.requires_grad = False

        if i <= block[1] and i >= block[0]:
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)  # 定义优化器

    train(model, train_loader, EPOCHS, optimizer, device)

    model.save_pretrained(Next_model)
    tokenizer.save_pretrained(Next_model)

