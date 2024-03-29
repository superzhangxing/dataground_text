# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy
from scipy.sparse import lil_matrix, dok_matrix, csr_matrix
import heapq


# 统计在词的df（文档频率）
def stat_df(docs):
    df_dict = {}
    for doc in tqdm(docs):
        word_list = doc.split()
        word_set = set(word_list)
        for word in word_set:
            df_dict[word] = df_dict.get(word, 0) + 1
    df_list = sorted(df_dict.items(), key=lambda d:d[1], reverse=True)
    return df_list, df_dict

# 统计词的类率
def stat_cf(docs, labels):
    cf_dict = {}
    word_set_by_label = {}
    label_set = set(labels)
    for label in label_set:
        word_set_by_label[label] = set()
    for id, (doc, label) in tqdm(enumerate(zip(docs, labels))):
        word_list = doc.split()
        for word in word_list:
            word_set_by_label[label].add(word)
    for word_set in tqdm(word_set_by_label.values()):
        for word in word_set:
            cf_dict[word] = cf_dict.get(word, 0) + 1
    cf_list = sorted(cf_dict.items(), key=lambda d:d[1], reverse=True)
    return cf_list, cf_dict

# 建立映射
def build_map(docs, labels):
    label_set = set(labels)
    word_set = set()
    for doc in tqdm(docs):
        for word in doc.split():
            word_set.add(word)

    label2id = {}
    id2label = {}
    word2id = {}
    id2word = {}

    label_id = word_id = 0

    for label in tqdm(label_set):
        label2id[label] = label_id
        id2label[label_id] = label
        label_id += 1

    for word in tqdm(word_set):
        word2id[word] = word_id
        id2word[word_id] = word
        word_id += 1
    return word2id, id2word, label2id, id2label

def calculate_dc_bdc(docs, labels, word2id, label2id):
    num_label = len(label2id)
    num_word = len(word2id)
    assert num_label > 1
    tf_matrix = np.zeros(shape=(num_word, num_label), dtype=int)
    p_matrix = np.zeros(shape=(num_word, num_label), dtype=float)

    for i, (doc, label) in tqdm(enumerate(zip(docs, labels))):
        word_list = doc.split()
        label_id = label2id[label]
        for word in word_list:
            tf_matrix[word2id[word]][label_id] += 1

    tf_label = np.sum(tf_matrix, axis = 0)
    tf_word = np.sum(tf_matrix, axis = 1)
    p_matrix = tf_matrix / tf_label
    dc_matrix = (tf_matrix.T / tf_word).T
    dc_df = pd.DataFrame(dc_matrix)
    dc_df = dc_df.applymap(lambda x: x*np.log(x) if x>0. else x)
    dc_matrix = dc_df.values
    dc_array = np.sum(dc_matrix, axis=1)
    dc_array = 1. + dc_array/np.log(num_label)

    p_word = np.sum(p_matrix, axis=1)
    bdc_matrix = (p_matrix.T / p_word).T
    bdc_df = pd.DataFrame(bdc_matrix)
    bdc_df = bdc_df.applymap(lambda x: x*np.log(x) if x>0. else x)
    bdc_matrix = bdc_df.values
    bdc_array = np.sum(bdc_matrix, axis=1)
    bdc_array = 1. + bdc_array/np.log(num_label)
    return dc_array, bdc_array

def calculate_p_matrix(docs, labels, word2id, label2id):
    num_label = len(label2id)
    num_word = len(word2id)
    assert num_label > 1
    p_matrix = np.zeros(shape=(num_word, num_label), dtype=float)

    for i, (doc, label) in tqdm(enumerate(zip(docs, labels))):
        word_list = doc.split()
        label_id = label2id[label]
        length = len(word_list)
        if length < 1:
            continue
        temp = 1./length
        for word in word_list:
            p_matrix[word2id[word]][label_id] += temp
    p_label = np.sum(p_matrix, axis=0)
    p_matrix = p_matrix / p_label
    return p_matrix


def build_vocab(train_data, dev_data):
    data = []
    data.extend(train_data)
    data.extend(dev_data)
    word_set=set()
    for doc in data:
        for word in doc.split():
            word_set.add(word)
    word2id = {}
    id2word = {}
    id = 0
    for word in word_set:
        word2id[word] = id
        id2word[id] = word
        id += 1
    return word2id, id2word

def compute_tf_matrix(train_data, dev_data, test_data, word2id):
    train_matrix = lil_matrix((len(train_data), len(word2id)),dtype=float)
    dev_matrix = lil_matrix((len(dev_data), len(word2id)), dtype=float)
    test_matrix = lil_matrix((len(test_data), len(word2id)), dtype = float)
    id = 0
    for doc in tqdm(train_data):
        for word in doc.split():
            if word in word2id:
                train_matrix[id, word2id[word]] += 1
        id += 1
    id = 0
    for doc in tqdm(dev_data):
        for word in doc.split():
            if word in word2id:
                dev_matrix[id, word2id[word]] += 1
        id += 1
    id = 0
    for doc in tqdm(test_data):
        for word in doc.split():
            if word in word2id:
                test_matrix[id, word2id[word]] += 1
        id += 1
    return train_matrix.tocsr(), dev_matrix.tocsr(), test_matrix.tocsr()

def compute_dc_matrix(train_data, dev_data, test_data, dc_bdc_file, word2id):
    dc_dict = {}
    dc_data = pd.read_csv(dc_bdc_file, header=None, dtype={0:str, 1:float, 2:float})
    for index, row in dc_data.iterrows():
        dc_dict[row[0]] = row[1]

    train_matrix = lil_matrix((len(train_data), len(word2id)),dtype=float)
    dev_matrix = lil_matrix((len(dev_data), len(word2id)), dtype=float)
    test_matrix = lil_matrix((len(test_data), len(word2id)), dtype = float)

    id = 0
    for doc in tqdm(train_data):
        for word in set(doc.split()):
            if word in word2id:
                train_matrix[id, word2id[word]] = dc_dict[word]
        id += 1
    id = 0
    for doc in tqdm(dev_data):
        for word in set(doc.split()):
            if word in word2id:
                dev_matrix[id, word2id[word]] =dc_dict[word]
        id += 1
    id = 0
    for doc in tqdm(test_data):
        for word in set(doc.split()):
            if word in word2id:
                test_matrix[id, word2id[word]] = dc_dict[word]
        id += 1

    return train_matrix.tocsr(), dev_matrix.tocsr(), test_matrix.tocsr()


def compute_bdc_matrix(train_data, dev_data, test_data, dc_bdc_file, word2id):
    dc_dict = {}
    dc_data = pd.read_csv(dc_bdc_file, header=None, dtype={0:str, 1:float, 2:float})
    for index, row in dc_data.iterrows():
        dc_dict[row[0]] = row[2]

    train_matrix = lil_matrix((len(train_data), len(word2id)), dtype=float)
    dev_matrix = lil_matrix((len(dev_data), len(word2id)), dtype=float)
    test_matrix = lil_matrix((len(test_data), len(word2id)), dtype=float)

    id = 0
    for doc in tqdm(train_data):
        for word in set(doc.split()):
            if word in word2id:
                train_matrix[id, word2id[word]] = dc_dict[word]
        id += 1
    id = 0
    for doc in tqdm(dev_data):
        for word in set(doc.split()):
            if word in word2id:
                dev_matrix[id, word2id[word]] = dc_dict[word]
        id += 1
    id = 0
    for doc in tqdm(test_data):
        for word in set(doc.split()):
            if word in word2id:
                test_matrix[id, word2id[word]] = dc_dict[word]
        id += 1

    return train_matrix.tocsr(), dev_matrix.tocsr(), test_matrix.tocsr()


if __name__ == '__main__':
    train_data = pd.read_csv('../new_data/train_set.csv')
    docs = train_data['word_seg'].values.tolist()
    labels = train_data['class'].values.tolist()
    # print('start statistic df...')
    # df_list, df_dict = stat_df(docs)
    # with open('../analysis/df.csv', 'w', encoding='utf-8') as f:
    #     for df in df_list:
    #         f.write(df[0]+','+str(df[1]))
    #         f.write('\n')
    # print('start statistic cf...')
    # cf_list, cf_dict = stat_cf(docs, labels)
    # with open('../analysis/cf.csv', 'w', encoding='utf-8') as f:
    #     for cf in cf_list:
    #         f.write(cf[0]+','+str(cf[1]))
    #         f.write('\n')

    # df/cf
    # print('start statistic df_cf_rate...')
    # df_cf_rate_dict = {}
    # for word in df_dict.keys():
    #     rate = df_dict[word]//cf_dict[word]
    #     df_cf_rate_dict[word] = rate
    # df_cf_rate_list = sorted(df_cf_rate_dict.items(), key=lambda d:d[1], reverse=True)
    # with open('../analysis/df_cf_rate.csv', 'w', encoding='utf-8') as f:
    #     for rate in df_cf_rate_list:
    #         f.write(rate[0]+','+str(rate[1])+','+str(df_dict[rate[0]])+','+str(cf_dict[rate[0]]))
    #         f.write('\n')

    # dc, bdc
    # word2id, id2word, label2id, id2label = build_map(docs, labels)
    # dc_array, bdc_array = calculate_dc_bdc(docs, labels, word2id, label2id)
    # f = open('../analysis/dc_bdc', 'w', encoding='utf-8')
    # for i, (dc, bdc) in tqdm(enumerate(zip(dc_array, bdc_array))):
    #     w_str = str(id2word[i])+','+str(dc)+','+str(bdc)+'\n'
    #     f.write(w_str)
    # f.close()

    # train_data = pd.read_csv('../preprocess_data/train_word_seg', header=None, squeeze=True).values.tolist()
    # dev_data = pd.read_csv('../preprocess_data/dev_word_seg', header=None, squeeze=True).values.tolist()
    # test_data = pd.read_csv('../preprocess_data/test_word_seg', header=None, squeeze=True).values.tolist()
    # word2id, id2word = build_vocab(train_data, dev_data)
    # train_matrix, dev_matrix, test_matrix = compute_dc_matrix(train_data, dev_data, test_data, '../analysis/dc_bdc', word2id)

    # statistic p matrix
    word2id, id2word, label2id, id2label = build_map(docs, labels)
    p_matrix = calculate_p_matrix(docs, labels, word2id, id2word)
    n = 100
    p_max_matrix = np.argsort(p_matrix, axis=0)[p_matrix.shape[0]-n:]
    f = open('../analysis/p_max', 'w', encoding='utf-8')
    for i in p_max_matrix.shape[0]:
        for j in p_max_matrix.shape[1]:
            f.write(id2word[p_max_matrix[i,j]]+',')
        f.write('\n')




# filter word: df/cf>3