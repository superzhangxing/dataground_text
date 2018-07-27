from src.model import ModelTemplate
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import time

class ModelLightGBM(ModelTemplate):
    def __init__(self, num_class, label_shift, X_train, y_train, X_dev=None, y_dev=None):
        super(ModelLightGBM, self).__init__(num_class, label_shift, X_train, y_train, X_dev, y_dev)
        self.params = {
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
            'num_threads': 7
        }
        self.gbm = None

    def eval_f1(self, preds, train_data):
        labels = train_data.get_label()
        predict_labels = np.reshape(preds, [self.num_class, preds.size // self.num_class])
        predict_labels = np.argmax(predict_labels, axis=0)
        F1 = f1_score(labels, predict_labels, average='macro')
        return 'F1', F1, True


    def train(self):
        train_dataset = lgb.Dataset(self.X_train, self.y_train)
        dev_dataset = lgb.Dataset(self.X_dev, self.y_dev)
        self.gbm = lgb.train(self.params, train_dataset, 500, valid_sets=dev_dataset, feval=self.eval_f1,
                  early_stopping_rounds=10)

    def predict(self, X_test):
        y_test = self.gbm.predict(X_test)
        y_test = np.argmax(y_test, axis=-1)  # lgb predict label from 0 to 18
        if self.label_shift != 0:
            y_test -= self.label_shift  # label from 1 to 19
        return y_test

    def save_model(self, file):
        buildTime = '-'.join(time.asctime(time.localtime(time.time())).strip().split(' ')[1:-1])
        buildTime = '-'.join(buildTime.split(':'))
        self.gbm.save_model(file+'-'+buildTime)
