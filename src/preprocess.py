import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# split train data to train dev data
def split(source_file, train_file, dev_file):
    source_data = pd.read_csv(source_file)
    train_data, dev_data = train_test_split(source_data, test_size = 0.1)
    train_data.to_csv(train_file)
    dev_data.to_csv(dev_file)

# statistic word/char average length
def stat_avg_len(data_file):
    data = pd.read_csv(data_file)
    article_list = data['article'].values.tolist()
    word_seg_list = data['word_seg'].values.tolist()
    article_count_list = [len(article.split()) for article in tqdm(article_list)]
    word_seg_count_list = [len(word_seg.split()) for word_seg in tqdm(word_seg_list)]
    article_avg_len = np.mean(np.array(article_count_list))
    word_seg_avg_len = np.mean(np.array(word_seg_count_list))

    return article_avg_len, word_seg_avg_len

# statistic number of samples in each class
def stat_class_sample_number(train_file):
    train_data = pd.read_csv(train_file)
    label_list = train_data['class'].values.tolist()
    label_array = np.array([int(label)+1 for label in label_list])
    label_num = np.bincount(label_array)
    return label_num


# 截断每行的词
def cut_data(origin_data_file, cut_data_file, cut_num = 500, cut_type='front'):
    """
    :param origin_data_file:
    :param cut_data_file:
    :param cut_num:
    :param cut_type:  'front', 'mid', 'back'
    :return:
    """
    origin_data = pd.read_csv(origin_data_file, header=None)
    origin_data_list = origin_data.values.tolist()
    f_cut = open(cut_data_file, 'w', encoding='utf-8')
    for origin_data_line in tqdm(origin_data_list):
        origin_data_split = origin_data_line[0].split()
        length = len(origin_data_split)
        if length <= cut_num:
            cut_data_line = origin_data_line[0]
        else:
            if cut_type == 'front':
                cut_data_line = ' '.join(origin_data_split[:cut_num])
            elif cut_type == 'mid':
                cut_data_line = ' '.join(origin_data_split[(length-cut_num)//2:(length+cut_num)//2])
            else:
                cut_data_line = ' '.join(origin_data_split[length-cut_num:])
        f_cut.write(cut_data_line)
        f_cut.write('\n')
    f_cut.close()


if __name__ == '__main__':
    # split('../new_data/train_set.csv', '../preprocess_data/train', '../preprocess_data/dev')
    #
    # train_data = pd.read_csv('../preprocess_data/train')
    # dev_data = pd.read_csv('../preprocess_data/dev')
    # test_data = pd.read_csv('../preprocess_data/test')
    # train_data['article'].to_csv('../preprocess_data/train_article', header=False, index=False, mode='w', sep='\n')
    # dev_data['article'].to_csv('../preprocess_data/dev_article', header=False, index=False, mode='w', sep='\n')
    # test_data['article'].to_csv('../preprocess_data/test_article', header=False, index=False, mode='w', sep='\n')
    # train_data['word_seg'].to_csv('../preprocess_data/train_word_seg', header=False, index=False, mode='w', sep='\n')
    # dev_data['word_seg'].to_csv('../preprocess_data/dev_word_seg', header=False, index=False, mode='w', sep='\n')
    # test_data['word_seg'].to_csv('../preprocess_data/test_word_seg', header=False, index=False, mode='w', sep='\n')
    # train_data['class'].to_csv('../preprocess_data/train_class', header=False, index=False, mode='w', sep='\n')
    # dev_data['class'].to_csv('../preprocess_data/dev_class', header=False, index=False, mode='w', sep='\n')
    # cut_data('../preprocess_data/train_article', '../preprocess_data/train_article_cut_500_front', cut_num=500, cut_type='front')
    # cut_data('../preprocess_data/train_word_seg', '../preprocess_data/train_word_seg_cut_500_front', cut_num=500, cut_type='front')
    # cut_data('../preprocess_data/dev_article', '../preprocess_data/dev_article_cut_500_front', cut_num=500, cut_type='front')
    # cut_data('../preprocess_data/dev_word_seg', '../preprocess_data/dev_word_seg_cut_500_front', cut_num=500, cut_type='front')
    # cut_data('../preprocess_data/test_article', '../preprocess_data/test_article_cut_500_front', cut_num=500, cut_type='front')
    # cut_data('../preprocess_data/test_word_seg', '../preprocess_data/test_word_seg_cut_500_front', cut_num=500, cut_type='front')

    # train_article_avg_len, train_word_seg_avg_len = stat_avg_len('../new_data/train_set.csv')
    # test_article_avg_len, test_word_seg_avg_len = stat_avg_len('../new_data/test_set.csv')
    # print('train article len: %d, train word seg len: %d' % (train_article_avg_len, train_word_seg_avg_len))
    # print('test article len: %d, test word seg len: %d' % (test_article_avg_len, test_word_seg_avg_len))

    label_num = stat_class_sample_number('../new_data/train_set.csv')
    print(label_num)
