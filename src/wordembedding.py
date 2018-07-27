# -*- coding: utf-8 -*-

import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.word2vec import Text8Corpus
from gensim.models.word2vec import LineSentence
import os



# ----对于比赛任务----
# 读数据
# article_file_name = 'article'
# word_seg_file_name = 'word_seg'
# article_f = open(os.path.join('new_data',article_file_name), 'w', encoding='utf-8')
# word_seg_f = open(os.path.join('new_data', word_seg_file_name), 'w', encoding='utf-8')
# with open('new_data/train_set.csv', 'r', encoding='utf-8') as f:
#   for id, line in enumerate(f):
#     if id==0:
#       continue
#     if id % 10000 == 0:
#       print('process line :%d' % (id))
#     line_split = line.split(',')
#     article = line_split[1]
#     word_seg = line_split[2]
#     article_f.write(article)
#     article_f.write('\n')
#     word_seg_f.write(word_seg)
#     word_seg_f.write('\n')
#
# with open('new_data/test_set.csv', 'r', encoding='utf-8') as f:
#   for id, line in enumerate(f):
#     if id==0:
#       continue
#     if id % 10000 == 0:
#       print('process line :%d' % (id))
#     line_split = line.split(',')
#     article = line_split[1]
#     word_seg = line_split[2]
#     article_f.write(article)
#     article_f.write('\n')
#     word_seg_f.write(word_seg)
#     word_seg_f.write('\n')
# article_f.close()
# word_seg_f.close()
  
# 训练词向量
# sentences = LineSentence('new_data/article')
# model = Word2Vec(sentences,min_count=5, size=200,window=5, negative=5, sg=1,
#                 hs=0, iter=1, workers=4 )
# model.save("article_word2vec.model")
# 保存词典
#model.wv.save_word2vec_format('article_vec','article_vocab')

# sentences = LineSentence('new_data/word_seg')
# model = Word2Vec(sentences,min_count=5, size=200,window=5, negative=5, sg=1,
#                  hs=0, iter=1, workers=4 )
# model.save("word_seg_word2vec.model")
# model.get_latest_training_loss()
# 保存词典
#model.wv.save_word2vec_format('word_seg_vec','word_seg_vocab')

# 相似性搜索



# ---- text8 文件做示例----

# 加载数据，按照要求的格式
sentences = Text8Corpus('new_data/text8')
# 创建模型并训练
model = Word2Vec(sentences,size=200, window=5, min_count=5, workers=4, sg=1,
                hs=0, negative=5,iter=1 )
# 保存模型
model.save("text8_word2vec.model")
# 查看某个单词的向量
print(model.wv['computer'])
# 相似性搜索，测试效果
print(model.wv.most_similar('computer'))



# ----训练比赛数据----

# ----准备训练数据----
# 一行一个文本，存储
train_data = pd.read_csv('../new_data/train_set.csv')
test_data = pd.read_csv('../new_data/test_set.csv')
train_data['article'].to_csv('../new_data/article', header=False, index=False, mode='w', sep='\n')
test_data['article'].to_csv('../new_data/article', header=False, index=False, mode='a', sep='\n')
train_data['word_seg'].to_csv('../new_data/word_seg', header=False, index=False, mode='w', sep='\n')
test_data['word_seg'].to_csv('../new_data/word_seg', header=False, index=False, mode='a', sep='\n')

# ----训练article词向量----
# 加载数据，按照要求的格式
sentences = LineSentence('../new_data/article')
# 创建模型并训练
model = Word2Vec(sentences,size=200, window=5, min_count=5, workers=4, sg=1,
                hs=0, negative=5,iter=1 )
# 保存模型
model.save("article_word2vec.model")
# 保存词向量和词典文件
model.wv.save_word2vec_format('article_vec','article_vocab')

# ----训练word_seg词向量----
# 加载数据，按照要求的格式
sentences = LineSentence('../new_data/word_seg')
# 创建模型并训练
model = Word2Vec(sentences,size=200, window=5, min_count=5, workers=4, sg=1,
                hs=0, negative=5,iter=1 )
# 保存模型
model.save("word_seg_word2vec.model")
# 保存词向量和词典文件
model.wv.save_word2vec_format('word_seg_vec','word_seg_vocab')