# -*-coding:utf-8-*-
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def read_data(dir):

    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            dataset.append(line)

    return dataset


def extract_information(information):
    names = []
    organization = []
    location = []
    for list in information:
        name = ''
        organ = ''
        loc = ''
        for dic in list:
            if dic['entity'] == 'B-PER' or dic['entity'] == 'I-PER':
                if dic['word'].startswith('#'):
                    name = name.strip(' ') + dic['word'].strip('#')
                else:
                    name = name + ' ' + dic['word']
            if dic['entity'] == 'B-ORG' or dic['entity'] == 'I-ORG':
                if dic['word'].startswith('#'):
                    organ = organ.strip(' ') + dic['word'].strip('#')
                else:
                    organ = organ + ' ' + dic['word']
            if dic['entity'] == 'B-LOC' or dic['entity'] == 'I-LOC':
                if dic['word'].startswith('#'):
                    loc = loc.strip(' ') + dic['word'].strip('#')
                else:
                    loc = loc + ' ' + dic['word']
        if len(name) != 0:
            names.append(name.lstrip())
        if len(organ) != 0:
            organization.append(organ.lstrip())
        if len(loc) != 0:
            location.append(loc.lstrip())

    return names, organization, location


def extract_information_ner(information):
    names = []
    organization = []
    location = []

    for list in information:
        name = ''
        organ = ''
        loc = ''
        for dic in list:
            if dic['entity'] == 'B-PER' or dic['entity'] == 'I-PER':
                if dic['word'].startswith('#'):
                    name = name.strip(' ') + dic['word'].strip('#')
                else:
                    name = name + ' ' + dic['word']
            if dic['entity'] == 'B-ORG' or dic['entity'] == 'I-ORG':
                if dic['word'].startswith('#'):
                    organ = organ.strip(' ') + dic['word'].strip('#')
                else:
                    organ = organ + ' ' + dic['word']
            if dic['entity'] == 'B-LOC' or dic['entity'] == 'I-LOC':
                if dic['word'].startswith('#'):
                    loc = loc.strip(' ') + dic['word'].strip('#')
                else:
                    loc = loc + ' ' + dic['word']
        if len(name) != 0:
            names.append(name.lstrip())
        else:
            names.append("#")
        if len(organ) != 0:
            organization.append(organ.lstrip())
        else:
            organization.append("#")
        if len(loc) != 0:
            location.append(loc.lstrip())
        else:
            location.append("#")

    return names, organization, location


def extract_name(save_path, name_save_path):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("./bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    data = read_data(save_path + '/zip_accept.txt')

    #data = read_data(save_path)

    ner_results = nlp(data)

    names, organization, location = extract_information(ner_results)

    with open(name_save_path + '/name.txt', 'w+') as f:
        for name in names:
            f.write(name + '\n')

    with open(name_save_path + '/organization.txt', 'w+') as f:
        for organ in organization:
            f.write(organ + '\n')

    with open(name_save_path + '/location.txt', 'w+') as f:
        for loc in location:
            f.write(loc + '\n')


def extract_name_data(save_path, name_save_path):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("./bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    data = read_data(save_path)

    ner_results = nlp(data)

    names, organization, location = extract_information(ner_results)

    with open(name_save_path + '/name.txt', 'w+') as f:
        for name in names:
            f.write(name + '\n')

    with open(name_save_path + '/organization.txt', 'w+') as f:
        for organ in organization:
            f.write(organ + '\n')

    with open(name_save_path + '/location.txt', 'w+') as f:
        for loc in location:
            f.write(loc + '\n')


