import json

def find_dataset(path):
	with open(path) as file:
		data = json.load(file)
		return data

training_sentences = []
training_labels = []
labels = []
responses = []

def prep_data(path):
	"""
	:returns: training_sentences, training_labels, responses(list), labels, num_classes
	"""
	data = find_dataset(path=path)
  
	for intent in data['intents']:
		for pattern in intent['patterns']:
			training_sentences.append(pattern)
			training_labels.append(intent['tag'])
			responses.append(intent['responses'])
     
		if intent['tag'] not in labels:
			labels.append(intent['tag'])
			
		num_classes = len(labels)
		return training_sentences, training_labels, responses, labels, num_classes, data
            

#training_sentences, training_labels, responses, labels, num_classes = prep_data()

def add_reddit_data(corpus):
	pass

"""
print(training_sentences[0])
print(training_labels[0])
print(responses[0][0])
print(labels[0])
print(num_classes)
"""
