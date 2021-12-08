import json

import numpy as np 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

from config import *

class Model:
    def __init__(self, num_classes):

        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        self.model.add(tf.keras.layers.Flatten()) 
        self.model.add(Dense(32, activation='elu'))
        self.model.add(Dense(16, activation='elu'))
        self.model.add(Dense(num_classes, activation='softmax'))

    def return_model(self):
        return self.model

