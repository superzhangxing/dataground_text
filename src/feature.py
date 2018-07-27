from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import corpora
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
import time
import pandas as pd
from tqdm import tqdm

class Feature_Tfidf(object):
    def __init__(self, X_train_org, X_dev_org, X_test_org):
        self.vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, use_idf=True, smooth_idf=True, sublinear_tf=False)
        self.X_train_org = X_train_org
        self.X_dev_org = X_dev_org
        self.X_test_org = X_test_org

    def transform(self):
        X_train = self.vectorizer.fit_transform(self.X_train_org)
        X_dev = self.vectorizer.transform(self.X_dev_org)
        X_test = self.vectorizer.transform(self.X_test_org)
        return X_train, X_dev, X_test

class Feature_Doc2Vec(object):
    def __init__(self, X_train_org, X_dev_org, X_test_org):
        self.X_train_org = X_train_org
        self.X_dev_org = X_dev_org
        self.X_test_org = X_test_org
        self.model = None

    def train(self):
        corpus = []
        corpus.extend(self.X_train_org)
        corpus.extend(self.X_dev_org)
        corpus.extend(self.X_test_org)
        corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
        self.model = Doc2Vec(vector_size=300, dm = 1, min_count=3, epochs=20)
        self.model.build_vocab(corpus)
        print('Start training doc2vec...')
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def transform(self):
        train_len = len(self.X_train_org)
        dev_len = len(self.X_dev_org)
        test_len = len(self.X_test_org)


    def save_model(self, file):
        buildTime = '-'.join(time.asctime(time.localtime(time.time())).strip().split(' ')[1:-1])
        buildTime = '-'.join(buildTime.split(':'))
        self.model.save(file+'-'+buildTime)
        self.model.docvecs.save_word2vec_format(file+'_docvecs')


# class Feature_LDA(object):
#     def __init__(self, X_train_org, X_dev_org, X_test_org):

if __name__ == '__main__':
    # X_train_word_seg = pd.read_csv('../preprocess_data/train_word_seg_cut_500_front', header=None,
    #                                squeeze=True).values.tolist()
    # y_train = pd.read_csv('../preprocess_data/train_class', header=None, squeeze=True).values.tolist()
    # y_train = [int(y) for y in y_train]
    # X_dev_word_seg = pd.read_csv('../preprocess_data/dev_word_seg_cut_500_front', header=None,
    #                              squeeze=True).values.tolist()
    # y_dev = pd.read_csv('../preprocess_data/dev_class', header=None, squeeze=True).values.tolist()
    # y_dev = [int(y) for y in y_dev]
    # X_test_word_seg = pd.read_csv('../preprocess_data/test_word_seg_cut_500_front', header=None,
    #                               squeeze=True).values.tolist()
    # corpus_word_seg = []
    # corpus_word_seg.extend(X_train_word_seg)
    # corpus_word_seg.extend(X_dev_word_seg)
    # corpus_word_seg.extend(X_test_word_seg)
    # corpus_word_seg = [line.split() for line in tqdm(corpus_word_seg)]
    # dictionary_word_seg = corpora.Dictionary(corpus_word_seg)
    # dictionary_word_seg.save('../topic_model/dictionary_word_seg')
    # corpus = [dictionary_word_seg.doc2bow(text) for text in corpus_word_seg]
    # corpora.MmCorpus.serialize('../topic_model/corpus_word_seg', corpus)

    dictionary_word_seg  = corpora.Dictionary.load('../topic_model/dictionary_word_seg')
   # dictionary_word_seg.save_as_text('../topic_model/dictionary_word_seg.txt')
   #  corpus_word_seg = corpora.MmCorpus('../topic_model/corpus_word_seg')
    # lsi = LsiModel(corpus=corpus_word_seg, id2word=dictionary_word_seg, num_topics=400)

    # tf-idf
    # corpus_train_word_seg = [line.split() for line in tqdm(X_train_word_seg)]
    # corpus_train_word_seg = [dictionary_word_seg.doc2bow(text) for text in corpus_train_word_seg]
    # corpus_dev_word_seg = [line.split() for line in tqdm(X_dev_word_seg)]
    # corpus_dev_word_seg = [dictionary_word_seg.doc2bow(text) for text in corpus_dev_word_seg]
    # corpus_test_word_seg = [line.split() for line in tqdm(X_test_word_seg)]
    # corpus_test_word_seg = [dictionary_word_seg.doc2bow(text) for text in corpus_test_word_seg]
    # corpus_train_dev_word_seg = []
    # corpus_train_dev_word_seg.extend(corpus_train_word_seg)
    # corpus_train_dev_word_seg.extend(corpus_dev_word_seg)
    # model = TfidfModel(corpus_train_dev_word_seg)
    # model.save('../topic_model/train_dev_word_seg_tfidf_model')
    # corpus_train_word_seg_tfidf = model.__getitem__(corpus_train_word_seg)
    # corpora.MmCorpus.serialize('../topic_model/corpus_train_word_seg_tfidf', corpus_train_word_seg_tfidf)
    # corpus_dev_word_seg_tfidf = model.__getitem__(corpus_dev_word_seg)
    # corpora.MmCorpus.serialize('../topic_model/corpus_dev_word_seg_tfidf', corpus_dev_word_seg_tfidf)
    # corpus_test_word_seg_tfidf = model.__getitem__(corpus_test_word_seg)
    # corpora.MmCorpus.serialize('../topic_model/corpus_test_word_seg_tfidf', corpus_test_word_seg_tfidf)

    corpus_train_word_seg_tfidf = corpora.MmCorpus('../topic_model/corpus_train_word_seg_tfidf')
    corpus_dev_word_seg_tfidf = corpora.MmCorpus('../topic_model/corpus_dev_word_seg_tfidf')
    corpus_test_word_seg_tfidf = corpora.MmCorpus('../topic_model/corpus_test_word_seg_tfidf')
    corpus_word_seg_tfidf = []
    corpus_word_seg_tfidf.extend(corpus_train_word_seg_tfidf)
    corpus_word_seg_tfidf.extend(corpus_dev_word_seg_tfidf)
    corpus_word_seg_tfidf.extend(corpus_test_word_seg_tfidf)


    # lsi
    print('Start train lsi...')
    lsi_model = LsiModel(corpus=corpus_word_seg_tfidf, id2word=dictionary_word_seg, num_topics=400)
    lsi_model.save('../topic_model/word_seg_lsi_model')
    corpus_train_word_seg_lsi = lsi_model[corpus_train_word_seg_tfidf]
    corpus_dev_word_seg_lsi = lsi_model[corpus_dev_word_seg_tfidf]
    corpus_test_word_seg_lsi = lsi_model[corpus_test_word_seg_tfidf]
    corpora.MmCorpus.serialize('../topic_model/corpus_train_word_seg_lsi', corpus_train_word_seg_lsi)
    corpora.MmCorpus.serialize('../topic_model/corpus_dev_word_seg_lsi', corpus_dev_word_seg_lsi)
    corpora.MmCorpus.serialize('../topic_model/corpus_test_word_seg_lsi', corpus_test_word_seg_lsi)

    #lda
    print('Start train lda...')
    lda_model = LdaModel(corpus=corpus_word_seg_tfidf, id2word=dictionary_word_seg, num_topics=100, update_every=1,
                         chunksize=1000, passes=1)
    lda_model.save('../topic_model/word_seg_lda_model')
    corpus_train_word_seg_lda = lda_model[corpus_train_word_seg_tfidf]
    corpus_dev_word_seg_lda = lda_model[corpus_dev_word_seg_tfidf]
    corpus_test_word_seg_lda = lda_model[corpus_test_word_seg_tfidf]
    corpora.MmCorpus.serialize('../topic_model/corpus_train_word_seg_lda', corpus_train_word_seg_lda)
    corpora.MmCorpus.serialize('../topic_model/corpus_dev_word_seg_lda', corpus_dev_word_seg_lda)
    corpora.MmCorpus.serialize('../topic_model/corpus_test_word_seg_lda', corpus_test_word_seg_lda)