# -*- coding:utf8 -*-
import pandas as pd
from src.feature import Feature_Tfidf, Feature_Doc2Vec
from src.model_lightgbm import ModelLightGBM

# load data
print('Start Load file...')
X_train_word_seg = pd.read_csv('../preprocess_data/train_word_seg_cut_500_front', header=None, squeeze=True).values.tolist()
y_train = pd.read_csv('../preprocess_data/train_class', header=None, squeeze=True).values.tolist()
y_train = [int(y) for y in y_train]
X_dev_word_seg = pd.read_csv('../preprocess_data/dev_word_seg_cut_500_front', header=None, squeeze=True).values.tolist()
y_dev = pd.read_csv('../preprocess_data/dev_class', header=None, squeeze=True).values.tolist()
y_dev = [int(y) for y in y_dev]
X_test_word_seg = pd.read_csv('../preprocess_data/test_word_seg_cut_500_front', header=None, squeeze=True).values.tolist()

# feature

# tf-idf
# print('Start generate feature...')
# feature_tfidf = Feature_Tfidf(X_train_word_seg, X_dev_word_seg, X_test_word_seg)
# X_train, X_dev, X_test = feature_tfidf.transform()

# doc2vec
print('Start generate feature...')
feature_docvec = Feature_Doc2Vec(X_train_word_seg, X_dev_word_seg, X_test_word_seg)
feature_docvec.train()

# model
print('Start train...')
model_lightgbm = ModelLightGBM(num_class=19, label_shift=-1, X_train=X_train, y_train=y_train,
                               X_dev = X_dev, y_dev = y_dev)
model_lightgbm.train()
model_lightgbm.save_model('../lightgbm/model')
y_test = model_lightgbm.predict(X_test)

# save predict file
with open('../lightgbm/predict.csv', 'w', encoding='utf-8') as f:
    f.write('id,class')
    f.write('\n')
    for id in range(len(y_test)):
        f.write(str(id))
        f.write(',')
        f.write(str(y_test[id]))
        f.write('\n')