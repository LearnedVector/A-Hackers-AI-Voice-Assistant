from flask.wrappers import Response
import joblib
import torch
import flask
from flask import Flask
from flask import request
import sys
sys.path.append('./neuralnet')
from neuralnet.model import NLUModel  
from neuralnet import config
#TO DO: produce intent json
#TO DO: produce scenario json


app = Flask(__name__)

META_DATA =joblib.load('meta_data.bin')

ENC_ENTITY = META_DATA['enc_entity']
ENC_INTENT = META_DATA['enc_intent']
ENC_SCENARIO = META_DATA['enc_scenario']

NUM_ENTITY = len(ENC_ENTITY.classes_)
NUM_INTENT = len(ENC_INTENT.classes_)
NUM_SCENARIO = len(ENC_SCENARIO.classes_)


DEVICE = config.DEVICE
MODEL = NLUModel(NUM_ENTITY,NUM_INTENT,NUM_SCENARIO)
MODEL.load_state_dict(torch.load(config.MODEL_PATH,
                                 map_location= lambda storage, loc:storage))
MODEL.to(DEVICE)
MODEL = MODEL.eval()

def sentence_prediction(sentence):
    """Given a sentence it will return NLU model prediction for entity,intent and scenario, wrapped as  
    json object.

    Args:
        sentence ([str]): [Input string]

    Returns:
        words_labels [dict]: [Dictionary of shape (words_pieces x prediction_labels) eg. {'wake':'O', '5':'time'...}]
        words_scores  [dict]: [Dictionary of shape (words_pieces x num_classes) eg. {'wake':[class1:score,......], '5':[class1:score,.....]...}]
        intent_sentence_labels [dict]: [Dictionary of shape (Num_sentences,), eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
        intent_class_scores [dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}]
        scenario_sentence_labels [dict]:  [Dictionary of shape (Num_sentences,) eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
        scenario_class_scores[dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}
    """
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    sentence = str(sentence)
    sentence = " ".join(sentence.split())

    inputs = tokenizer.encode_plus(
        sentence,
        None,
        add_special_tokens=True,
        truncation=True,
        max_length = max_len
    )
    
    tokenized_ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    word_pieces = tokenizer.decode(inputs['input_ids']).split()[1:len(tokenized_ids)-1]

    #padding
    padding_len = max_len - len(tokenized_ids)
        
    ids = tokenized_ids + ([0] * padding_len)
    mask = mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)
    
    ids = torch.tensor(ids,dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    
    ids = ids.to(DEVICE,dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE,dtype=torch.long)
    ids = ids.to(DEVICE,dtype=torch.long)

    entity_hs,intent_hs,scenario_hs  = MODEL(ids=ids,mask=mask,token_type_ids=token_type_ids)
    
    entity_scores,entity_preds = to_yhat(entity_hs)
    entity_scores = entity_scores[1:len(tokenized_ids)-1, :]
    enitity_labels = ENC_ENTITY.inverse_transform(entity_preds)[1:len(tokenized_ids)-1]
    words_labels = words_to_labels( word_pieces , enitity_labels)
    words_scores = words_to_scores( word_pieces, entity_scores)

    

    intent_scores,intent_preds = to_yhat(intent_hs)
    intent_sentence_labels = sentence_to_labels(ENC_INTENT.inverse_transform(intent_preds), len(ids))
    intent_class_scores = classes_to_scores(ENC_INTENT.classes_,intent_scores)

    
    scenario_scores,scenario_preds = to_yhat(scenario_hs)
    scenario_sentence_labels = sentence_to_labels(ENC_SCENARIO.inverse_transform(scenario_preds),len(ids) )
    scenario_class_scores = classes_to_scores(ENC_SCENARIO.classes_,scenario_scores)
    
    return words_labels, words_scores, intent_sentence_labels,intent_class_scores, scenario_sentence_labels,scenario_class_scores


def classes_to_scores(classes,intent_scores):
    dict = {'scores':[]}
    for scores in intent_scores:
        dict['scores'] += [{c:s for c,s in zip(classes,scores)}]
    return dict

def sentence_to_labels(labels,num_sentence):
    return {i:l for l,i in zip(labels, range(num_sentence))}
        

def words_to_labels(word_pieces,labels):
    return {w:l for w,l in zip(word_pieces,labels)}

def words_to_scores(words_pieces, scores):
    return {w:cs for w,cs in zip(words_pieces, scores)}


def to_yhat(logits):
    logits = logits.view(-1, logits.shape[-1]).cpu().detach()
    probs = torch.softmax(logits, dim=1)
    y_hat = torch.argmax(probs, dim=1)
    return probs.numpy(),y_hat.numpy() 

@app.route("/test")
def predict():
    print(request.args)
    sentence = request.args.get('sentence')
    words_labels,_,intent_sentence_labels,_,scenario_sentence_lables,_ = sentence_prediction(sentence)
    # res = {'test':'hello world'}
    res = {}
    res['words_labels'] = words_labels
    return flask.jsonify(res)




if __name__ == "__main__":
    # sentence_prediction('wake me up at 5 am please')
    app.run(debug=True)