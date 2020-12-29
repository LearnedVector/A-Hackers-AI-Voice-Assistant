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

# META_DATA =joblib.load('meta_data.bin')

# ENC_ENTITY = META_DATA['enc_entity']
# ENC_INTENT = META_DATA['enc_intent']
# ENC_SCENARIO = META_DATA['enc_scenario']

# NUM_ENTITY = len(ENC_ENTITY.classes_)
# NUM_INTENT = len(ENC_INTENT.classes_)
# NUM_SCENARIO = len(ENC_SCENARIO.classes_)


# MODEL = NLUModel(NUM_ENTITY,NUM_INTENT,NUM_SCENARIO)
# MODEL.load_state_dict(torch.load(config.MODEL_PATH,
#                                  map_location= lambda storage, loc:storage))
# MODEL = torch.jit.load(config.TRACE_MODEL_PATH)
# MODEL.to(DEVICE)
# MODEL = MODEL.eval()
# DEVICE = config.DEVICE


class NLUEngine:
    def __init__(self):
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.device = config.DEVICE
        self.model = torch.jit.load(config.TRACE_MODEL_PATH).to(self.device).eval()
        
        self.meta_data =joblib.load('meta_data.bin')

        self.enc_entity = self.meta_data['enc_entity']
        self.enc_intent = self.meta_data['enc_intent']
        self.enc_scenario = self.meta_data['enc_scenario']

        self.num_entity = len(self.enc_entity.classes_)
        self.num_intent = len(self.enc_intent.classes_)
        self.num_scenario = len(self.enc_scenario.classes_)

    
    @staticmethod
    def classes_to_scores_json(classes,intent_scores):
        dict = {'scores':[]}
        for scores in intent_scores:
            dict['scores'] += [{c:s for c,s in zip(classes,scores)}]
        return dict

    @staticmethod
    def sentence_to_labels_json(labels,num_sentence):
        return {i:l for l,i in zip(labels, range(num_sentence))}
            

    @staticmethod
    def words_to_labels_json(word_pieces,labels):
        return {w:l for w,l in zip(word_pieces,labels)}

    @staticmethod
    def words_to_scores_json(words_pieces, scores):
        return {w:cs for w,cs in zip(words_pieces, scores)}


    @staticmethod
    def to_yhat(logits):
        logits = logits.view(-1, logits.shape[-1]).cpu().detach()
        probs = torch.softmax(logits, dim=1)
        y_hat = torch.argmax(probs, dim=1)
        return probs.numpy(),y_hat.numpy() 
    
    
    def process_sentence(self,sentence):
        sentence = str(sentence)
        sentence = " ".join(sentence.split())
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length = self.max_len
        )
        
        tokenized_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        word_pieces = self.tokenizer.decode(inputs['input_ids']).split()[1:-1] #the first token an the last token are special token

        #padding
        padding_len = self.max_len - len(tokenized_ids)
            
        ids = tokenized_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        
        ids = torch.tensor(ids,dtype=torch.long).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        return ids,mask,token_type_ids,tokenized_ids,word_pieces
    
    def sentence_prediction(self,ids,mask,token_type_ids):
        entity_hs,intent_hs,scenario_hs  = self.model(ids,mask,token_type_ids)
        return entity_hs,intent_hs,scenario_hs   
    
    def entity_extraction(self,entity_hs,word_pieces,tokenized_ids):
        entity_scores,entity_preds = to_yhat(entity_hs)
        entity_scores = entity_scores[1:len(tokenized_ids)-1, :]
        enitity_labels = self.enc_entity.inverse_transform(entity_preds)[1:len(tokenized_ids)-1]
        words_labels_json = self.words_to_labels_json( word_pieces , enitity_labels)
        words_scores_json = self.words_to_scores_json( word_pieces, entity_scores)
        return words_labels_json,words_scores_json
        
    def classification(self,logits,task='intent'):
        if task == 'intent':
            enc = self.enc_intent
        else:
            enc = self.enc_scenario
        class_scores,class_preds = to_yhat(logits)
        sentence_labels_json = self.sentence_to_labels_json(enc.inverse_transform(class_preds), len(class_preds))
        class_scores_json = self.classes_to_scores_json(enc.classes_,class_scores)
        return sentence_labels_json, class_scores_json
    
    def predict(self,sentence):
        ids,mask,token_type_ids,tokenized_ids,word_pieces = self.process_sentence(sentence)

        #forward the inputs throught the model and get logits
        entity_hs,intent_hs,scenario_hs = self.sentence_prediction(ids,mask,token_type_ids)

        #entity extraction
        words_labels_json,words_scores_json = self.entity_extraction(entity_hs,word_pieces,tokenized_ids)

        # intent and scenario classification
        intent_sentence_labels_json, intent_class_scores_json = self.classification(intent_hs,task='intent')
        scenario_sentence_labels_json, scenario_class_scores_json = self.classification(scenario_hs,task='scenario')
             
        return (words_labels_json, 
                words_scores_json, 
                intent_sentence_labels_json, 
                intent_class_scores_json, 
                scenario_sentence_labels_json, 
                scenario_class_scores_json)
        
@app.route("/test")
def predict(verbose=False):
    print(request.args)
    sentence = request.args.get('sentence')
    nlu_engine = NLUEngine()

    (words_labels, 
     words_scores, 
     intent_sentence_labels,
     intent_class_scores,
     scenario_sentence_labels,
     scenario_class_scores)= nlu_engine.predict(sentence)

    # res = {'test':'hello world'}
    res = {}
    res['words_labels'] = words_labels
    res['intent_sentence_labels'] = intent_sentence_labels
    res['scenario_sentence_labels'] = scenario_sentence_labels
    if verbose:
        res['words_scores'] = words_scores
        res['intent_class_scores'] = intent_class_scores
        res['scenario_class_scores'] = scenario_class_scores
    return flask.jsonify(res)




if __name__ == "__main__":
    # sentence_prediction('wake me up at 5 am please')
    app.run(debug=True)