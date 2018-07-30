# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

# 统计在词的df（文档频率）
def stat_df(docs):
    df_dict = {}
    for doc in docs:
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
    for id, (doc, label) in enumerate(zip(docs, labels)):
        word_list = doc.split()
        word_set_by_label[label] = word_set_by_label[label].union(set(word_list))
    for word_set in word_set_by_label.values():
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
    word2id, id2word, label2id, id2label = build_map(docs, labels)
    dc_array, bdc_array = calculate_dc_bdc(docs, labels, word2id, label2id)
    a = [1,2]



# filter word: df/cf>3