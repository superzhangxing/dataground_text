import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score

train = pd.read_csv('../preprocess_data/train_word_seg', header=None, squeeze=True)
train_label = pd.read_csv('../preprocess_data/train_class', header=None, squeeze=True)
dev = pd.read_csv('../preprocess_data/dev_word_seg', header=None, squeeze=True)
dev_label = pd.read_csv('../preprocess_data/dev_class', header=None, squeeze=True)
test = pd.read_csv('../preprocess_data/test_word_seg', header=None, squeeze=True)
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
train_term_doc = vec.fit_transform(train)
dev_term_doc = vec.transform(dev)
test_term_doc = vec.transform(test)
fid0=open('baseline.csv','w')

train_y=(train_label-1).astype(int)
dev_y = (dev_label-1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(train_term_doc,train_y)
dev_preds = lin_clf.predict(dev_term_doc)
test_preds = lin_clf.predict(test_term_doc)

dev_f1 = f1_score(dev_y, dev_preds, average='macro')
print('dev f1:%f' %dev_f1)

i=0
fid0.write("id,class"+"\n")
for item in test_preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()