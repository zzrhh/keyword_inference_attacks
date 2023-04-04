# -*-coding:utf-8-*-
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import time
from utilize.dirpath import generic_save_path, medical_head
import os
from tqdm import tqdm


BLOCK_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
PATHS = ['./data/private/train.txt']
num = 100000
target_save_path = medical_head
part = 'head'

def encode_string(paths, tokenizer, num_sens):
    string_tokenized = []
    for filename in paths:
        num_file = sum([1 for i in open(filename, "r")])
        print("开始处理...")
        start = time.time()
        with open(filename, 'r') as f:
            for idx, line in tqdm(enumerate(f), total=num_file):
                li = line.strip()
                lin = str(li).lstrip("BACKGROUND OBJECTIVE METHODS RESULTS CONCLUSIONS")
                lin = lin.strip()
                if lin != '' and lin[0:3] != '###':
                    lin = lin + tokenizer.eos_token
                    encode_idx = tokenizer.encode(lin)
                    string_tokenized.extend(encode_idx)

                if idx == num_sens:
                    print('sentences number:', num_sens)
                    break

        end = time.time()
    print("encode finish %s s" % (end - start))
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



# 模型目录
if not os.path.exists(target_save_path):
    os.makedirs(target_save_path)

# 载入预训练模型的分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained(generic_save_path)
model = GPT2LMHeadModel.from_pretrained(generic_save_path)

# 读取数据集

dataset = encode_string(PATHS, tokenizer, num)

inputs, labels = split_input_labels(dataset, BLOCK_SIZE)

# 构建数据集和数据迭代器，设定 batch_size 大小为 2
train_set = TensorDataset(inputs, labels)

train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print(device.type)

model.to(device)
model.train()

head = [2, 13]
block4 = [98, 145]
block8 = [50, 145]


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

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

train(model, train_loader, EPOCHS, optimizer, device)

model.save_pretrained(target_save_path)
tokenizer.save_pretrained(target_save_path)
