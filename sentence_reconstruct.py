# -*-coding:utf-8-*-
import torch
from torch.autograd import Variable
import random
from utilize.dirpath import target_save_path_sky
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utilize.dirpath import name_save_path, reconstruct_save_path
from tqdm import tqdm


def select_top_k(predictions, k=10):

    prediction_k = predictions[0, -1, :].sort(descending=True)[1][:k]
    predicted_index = random.choice(list(enumerate(prediction_k)))[0]

    return prediction_k[predicted_index].item()


def generate_sentence(model, device, tokenizer, generate_text, k=1, n=500):
    model.to(device)
    model.eval()

    total_predicted_text = generate_text

    indexed_tokens = tokenizer.encode(total_predicted_text)
    tokens_tensor = torch.tensor([indexed_tokens])

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


        indexed_tokens += [predicted_index]
        tokens_tensor = torch.tensor([indexed_tokens])

    return total_predicted_text


def read_data(dir):

    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            dataset.append(line)

    return dataset


def reconstruct_main(device, target_save_path, name_save_path, reconstruct_save_path):

    tokenizer_target = GPT2Tokenizer.from_pretrained(target_save_path)
    model_target = GPT2LMHeadModel.from_pretrained(target_save_path)

    names = read_data(name_save_path + '/name.txt')
    organizations = read_data(name_save_path + '/organization.txt')
    locations = read_data(name_save_path + '/location.txt')
    with open(reconstruct_save_path + '/name_reconstruct.txt', 'w+') as f:
        for name in tqdm(names):
            text = generate_sentence(model_target, device, tokenizer_target, name)
            f.write(text + '\n')

    with open(reconstruct_save_path + '/location_reconstruct.txt', 'w+') as f:
        for name in tqdm(locations):
            text = generate_sentence(model_target, device, tokenizer_target, name)
            f.write(text + '\n')


    with open(reconstruct_save_path + '/organization_reconstruct.txt', 'w+') as f:
        for name in tqdm(organizations):
            text = generate_sentence(model_target, device, tokenizer_target, name)
            f.write(text + '\n')
