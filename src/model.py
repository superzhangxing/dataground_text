import pandas as pd
import numpy as np
from abc import abstractmethod

class ModelTemplate(object):
    def __init__(self, num_class, label_shift, X_train, y_train, X_dev=None, y_dev=None):
        self.num_class = num_class
        self.label_shift = label_shift
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        if self.label_shift != 0:
            self.y_train = [label+self.label_shift for label in self.y_train]
            self.y_dev = [label+self.label_shift for label in self.y_dev]

    @abstractmethod
    def predict(self, X_test_file):
        pass

    @abstractmethod
    def predict_and_save_to_file(self, X_test_file, y_test_file):
        pass

    @abstractmethod
    def save_model(self, file):
        pass