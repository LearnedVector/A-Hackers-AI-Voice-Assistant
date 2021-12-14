import json
import requests


from VoiceAssistant.nlu_stable.client_config import *


class Client:
	def __init__(self):
		self.text = ""
		self.inference_list = []
		self.token = TOKEN
		self.head = HEAD
		pass

	def _ask_input(self, text):
		self.text = text
		self.inference_list = []
		print('text: ', self.text)
		self.raw_data = requests.get(f'{self.head}{self.token}/{self.text}')
		print(f'{self.head}{self.token}/{self.text}')
		self.data = json.loads(self.raw_data.text)
		self.inference_list = self.data
		return self.inference_list
		

	def _understand_intent(self, inference_list):
		self.inference_list = inference_list
		if self.inference_list[1] == "":
			return None
		else:
			return self.inference_list[1]

