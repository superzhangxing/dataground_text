# -*- coding:utf8 -*-
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import scipy
import numpy as np
from src.func import calculate_F1

NUM_CLASS = 19



# 准备数据
train_data = pd.read_csv('../new_data/train_set.csv')
test_data = pd.read_csv('../new_data/test_set.csv')
train_data['class'] = train_data['class'] - 1
train_data, dev_data = train_test_split(train_data, test_size=0.1)

train_data.to_csv('../new_data/train_set_split.csv')
dev_data.to_csv('../new_data/dev_set_split.csv')

train_article_list = train_data['article'].values.tolist()
train_word_seg_list = train_data['word_seg'].values.tolist()
train_label_list = train_data['class'].values.tolist()

dev_article_list = dev_data['article'].values.tolist()
dev_word_seg_list = dev_data['word_seg'].values.tolist()
dev_label_list = dev_data['class'].values.tolist()

test_article_list = test_data['article'].values.tolist()
test_word_seg_list = test_data['word_seg'].values.tolist()

# tf-idf
article_vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.9, use_idf=True, smooth_idf=True, sublinear_tf=False)
train_article_tf_idf = article_vectorizer.fit_transform(train_article_list)
dev_article_tf_idf = article_vectorizer.transform(dev_article_list)
test_article_tf_idf = article_vectorizer.transform(test_article_list)

word_seg_vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.9, use_idf=True, smooth_idf=True, sublinear_tf=False)
train_word_seg_tf_idf = word_seg_vectorizer.fit_transform(train_word_seg_list)
dev_word_seg_tf_idf = word_seg_vectorizer.transform(dev_word_seg_list)
test_word_seg_tf_idf = word_seg_vectorizer.transform(test_word_seg_list)

# lightgbm
lgb_article_train = lgb.Dataset(train_article_tf_idf, train_label_list)
lgb_article_dev = lgb.Dataset(dev_article_tf_idf, dev_label_list, reference=lgb_article_train)

lgb_word_seg_train = lgb.Dataset(train_word_seg_tf_idf, train_label_list)
lgb_word_seg_dev = lgb.Dataset(dev_word_seg_tf_idf, label=dev_label_list, reference=lgb_word_seg_train)

print('Start training ...')

# train
params = {
    'task': 'train',
    'objective': 'multiclass',
    'num_class': 19,
    'boosting': 'gbdt',
    'num_leaves': 100,
    'tree_learner': 'feature',
    'num_threads': 8,
    'metric': {'multi_logloss'},
    'is_provide_training_metric': 'true',
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'num_threads':7
}

def valid_F1(preds, train_data):
    labels = train_data.get_label()
    labels = [label+1 for label in labels]
    predict_labels = np.reshape(preds, [NUM_CLASS,preds.size//NUM_CLASS])
    predict_labels = np.argmax(predict_labels, axis=0)
    predict_labels += 1
    F1, F1_list = calculate_F1(labels, predict_labels.tolist())
    return 'F1', F1, True

gbm = lgb.train(params, lgb_word_seg_train, 500, valid_sets=lgb_word_seg_dev, feval=valid_F1, early_stopping_rounds=10)

print('Save model...')
gbm.save_model('../lightgbm/model.txt')


# predict
print('Predict...')
test_label = gbm.predict(test_word_seg_tf_idf)
test_label = np.argmax(test_label, axis=-1) # lgb predict label from 0 to 18
test_label += 1  # label from 1 to 19

# save predict
with open('../lightgbm/predict.csv', 'w', encoding='utf-8') as f:
    f.write('id,class')
    f.write('\n')
    for id in range(len(test_label)):
        f.write(str(id))
        f.write(',')
        f.write(str(test_label[id]))
        f.write('\n')



