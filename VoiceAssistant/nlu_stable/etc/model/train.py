import numpy as np 
import pickle
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from config import *
from data import data
from model import Model


training_sentences, training_labels, responses, labels, num_classes, data = data.prep_data(data_path)


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

class_model = Model(num_classes)

model = class_model.return_model()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

model.save('saved/model.h5')

pickle.dump(tokenizer, open('saved/tokenizer.pickle', 'wb'))
pickle.dump(lbl_encoder, open('saved/lbl_encoder.pickle', 'wb'))
pickle.dump(data, open('saved/data.pickle', 'wb'))

