import json
import pickle

import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model



from VoiceAssistant.nlu_stable.etc.model.config import * # hprams


class Brain:

	def __init__(self):
		super().__init__()

		self.model = load_model(model_path)
		self.text = ""
		self.inferred = []

		with open('nlu/etc/model/saved/tokenizer.pickle', 'rb') as handle:
			self.tokenizer = pickle.load(handle)

		with open('nlu/etc/model/saved/lbl_encoder.pickle', 'rb') as handle:
			self.lbl_encoder = pickle.load(handle)

		with open('nlu/etc/model/saved/data.pickle', 'rb') as handle:
			self.data = pickle.load(handle)

	
	def chat(self, text):
		self.inferred = []
		self.text = str(text).lower()

		self.result = self.model.predict(keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([self.text]),
	                                             truncating='post', maxlen=max_len))
		self.tag = self.lbl_encoder.inverse_transform([np.argmax(self.result)])

		for i in self.data['intents']:
			if i['tag'] == self.tag:
				self.inferred.append(np.random.choice(i['responses']))
				self.inferred.append(i['context_set'])
				break
		
		return self.inferred


	
	