# -*-coding:utf-8-*-
import re
import csv
from tqdm import tqdm
from utilize.dirpath import reconstruct_save_path_iter1,\
                            reconstruct_save_path_iter2, reconstruct_save_path_iter3,\
                            reconstruct_save_path_iter4, reconstruct_save_path_iter5, \
                            reconstruct_save_path_iter6, reconstruct_save_path_iter7, \
                            reconstruct_save_path_iter8, reconstruct_save_path_iter9, \
                            reconstruct_save_path_iter10
from bertNer import extract_information_ner
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from utilize.dirpath import name_save_path_iter1, name_save_path_iter2, name_save_path_iter3, \
                            name_save_path_iter4, name_save_path_iter5, name_save_path_iter6, \
                            name_save_path_iter7, name_save_path_iter8, name_save_path_iter9, \
                            name_save_path_iter10



def read_data(dir):

    dataset = []
    with open(dir, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            dataset.append(line)

    return dataset


def extract_sky_information(reconstruct_save_path):

    for i in range(len(reconstruct_save_path)):
        data = read_data(reconstruct_save_path[i] + '/name_reconstruct.txt')
        with open(reconstruct_save_path[i] + '/reconstruct.csv', "w+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "country", "airline", "date"])
            row = []
            for da in tqdm(data):
                try:
                    name = re.findall(r"(.+?) is an", da)[0]
                    country = re.findall(r"is an (.+?) who flew", da)[0]
                    airline = re.findall(r"flew on (.+?) in", da)[0]
                    date = re.findall(r"in (.+?) and says", da)[0]
                    tmp = []
                    tmp.append(name)
                    tmp.append(country)
                    tmp.append(airline)
                    tmp.append(date)
                    row.append(tmp)
                except IndexError:
                    continue
            for r in row:
                writer.writerow(r)

def extract_sky_information_ner(reconstruct_save_path):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("./bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    for i in range(len(reconstruct_save_path)):

        data_org = read_data(reconstruct_save_path[i] + '/organization_reconstruct.txt')
        data_name = read_data(reconstruct_save_path[i] + '/name_reconstruct.txt')
        data_loc = read_data(reconstruct_save_path[i] + '/location_reconstruct.txt')

        ner_results_org = nlp(data_org)
        ner_results_name = nlp(data_name)
        ner_results_loc = nlp(data_loc)

        names, organization, location = extract_information_ner(ner_results_org)
        names1, organization1, location1 = extract_information_ner(ner_results_name)
        names2, organization2, location2 = extract_information_ner(ner_results_loc)

        with open(reconstruct_save_path[i] + '/reconstruct_combine_ner.csv', "w+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "airline", "country", "date"])
            row = []
            i = 0
            for da in tqdm(data_org):
                try:
                    all = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", da)
                    review_date = all[0]
                    tmp = []
                    tmp.append(names[i])
                    tmp.append(organization[i])
                    tmp.append(location[i])
                    tmp.append(review_date)
                    row.append(tmp)
                except IndexError:
                    continue
                i = i + 1
            i = 0
            for da in tqdm(data_name):
                try:
                    all = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", da)
                    review_date = all[0]
                    tmp = []
                    tmp.append(names1[i])
                    tmp.append(organization1[i])
                    tmp.append(location1[i])
                    tmp.append(review_date)
                    row.append(tmp)
                except IndexError:
                    continue
                i = i + 1
            i = 0
            for da in tqdm(data_loc):
                try:
                    all = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", da)
                    review_date = all[0]
                    tmp = []
                    tmp.append(names2[i])
                    tmp.append(organization2[i])
                    tmp.append(location2[i])
                    tmp.append(review_date)
                    row.append(tmp)
                except IndexError:
                    continue
                i = i + 1

            for r in row:
                writer.writerow(r)

# def extract_movie_information(reconstruct_save_path):
#
#     for i in range(len(reconstruct_save_path)):
#         data = read_data(reconstruct_save_path[i] + '/organization_reconstruct.txt')
#         with open(reconstruct_save_path[i] + '/reconstruct.csv', "w+", newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["critic_name", "publisher_name", "review_date"])
#             row = []
#             for da in tqdm(data):
#                 try:
#                     all = re.findall(r"(.+?) in (\d{4}-\d{1,2}-\d{1,2}).(.+?) says", da)[0]
#                     publisher_name = all[0]
#                     review_date = all[1]
#                     critic_name = all[2]
#                     tmp = []
#                     tmp.append(critic_name)
#                     tmp.append(publisher_name)
#                     tmp.append(review_date)
#                     row.append(tmp)
#                 except IndexError:
#                     continue
#             for r in row:
#                 writer.writerow(r)
#
#
# def extract_movie_information_ner(reconstruct_save_path):
#     tokenizer = AutoTokenizer.from_pretrained("./bert-base-NER")
#     model = AutoModelForTokenClassification.from_pretrained("./bert-base-NER")
#
#     nlp = pipeline("ner", model=model, tokenizer=tokenizer)
#
#     for i in range(len(reconstruct_save_path)):
#
#         # na = read_data(name_save_path[i] + '/name.txt')
#         # org = read_data(name_save_path[i] + '/organization.txt')
#         # loc = read_data(name_save_path[i] + '/location.txt')
#
#         data_org = read_data(reconstruct_save_path[i] + '/organization_reconstruct.txt')
#         data_name = read_data(reconstruct_save_path[i] + '/name_reconstruct.txt')
#         data_loc = read_data(reconstruct_save_path[i] + '/location_reconstruct.txt')
#
#         ner_results_org = nlp(data_org)
#         ner_results_name = nlp(data_name)
#         ner_results_loc = nlp(data_loc)
#
#         names, organization, location = extract_information_ner(ner_results_org)
#         names1, organization1, location1 = extract_information_ner(ner_results_name)
#         names2, organization2, location2 = extract_information_ner(ner_results_loc)
#
#         with open(reconstruct_save_path[i] + '/reconstruct_combine_ner.csv', "w+", newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["critic_name", "publisher_name_or", "publisher_name_lo", "review_date"])
#             row = []
#             i = 0
#             for da in tqdm(data_org):
#                 try:
#                     all = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", da)
#                     review_date = all[0]
#                     tmp = []
#                     tmp.append(names[i])
#                     tmp.append(organization[i])
#                     tmp.append(location[i])
#                     tmp.append(review_date)
#                     row.append(tmp)
#                 except IndexError:
#                     continue
#                 i = i + 1
#             i = 0
#             for da in tqdm(data_name):
#                 try:
#                     all = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", da)
#                     review_date = all[0]
#                     tmp = []
#                     tmp.append(names1[i])
#                     tmp.append(organization1[i])
#                     tmp.append(location1[i])
#                     tmp.append(review_date)
#                     row.append(tmp)
#                 except IndexError:
#                     continue
#                 i = i + 1
#             i = 0
#             for da in tqdm(data_loc):
#                 try:
#                     all = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", da)
#                     review_date = all[0]
#                     tmp = []
#                     tmp.append(names2[i])
#                     tmp.append(organization2[i])
#                     tmp.append(location2[i])
#                     tmp.append(review_date)
#                     row.append(tmp)
#                 except IndexError:
#                     continue
#                 i = i + 1
#
#             for r in row:
#                 writer.writerow(r)


# path_new_our = '/sky_100000/limit_1'
# path_new_compare = '/sky_100000/limit_1/compare'
#
# reconstruct_save_path = [reconstruct_save_path_iter1, reconstruct_save_path_iter2, reconstruct_save_path_iter3,
#                          reconstruct_save_path_iter4, reconstruct_save_path_iter5, reconstruct_save_path_iter6,
#                          reconstruct_save_path_iter7, reconstruct_save_path_iter8, reconstruct_save_path_iter9,
#                          reconstruct_save_path_iter10]
# reconstruct_save_path = [path + path_new_our for path in reconstruct_save_path]
#
#
#
# extract_sky_information_ner(reconstruct_save_path)
# extract_sky_information(reconstruct_save_path)

