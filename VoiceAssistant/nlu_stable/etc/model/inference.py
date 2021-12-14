import json
import pickle
import numpy as np 
from tensorflow import keras
from tensorflow.keras.models import load_model


from config import *


model = load_model("saved/model.h5")

with open('saved/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('saved/lbl_encoder.pickle', 'rb') as handle:
    lbl_encoder = pickle.load(handle)

with open('saved/data.pickle', 'rb') as handle:
    data = pickle.load(handle)

def chat():
	print("Start messaging with the bot (type quit to stop)!\n")
	while True:
		print("User: ")
		inp = input()
		if inp.lower() == "quit":
			break

		result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
		tag = lbl_encoder.inverse_transform([np.argmax(result)])

		for i in data['intents']:
			if i['tag'] == tag:
				print("ChatBot:" + np.random.choice(i['responses']))
				print("intent asked for: ", i['context_set'], "\n")

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

chat()
