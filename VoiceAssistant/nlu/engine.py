import joblib
import torch
import flask
from flask import Flask
from flask import request
import sys
sys.path.append('./neuralnet')
from neuralnet.model import NLUModel  
from neuralnet import config
import argparse

app = Flask(__name__)

class NLUEngine:
    def __init__(self,model_path):
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.device = config.DEVICE
        self.model = torch.jit.load(model_path).to(self.device).eval()
        
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
        """ Given a sentence stirng it will return rquired inputs for NLU model's forward pass

        """
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
        """Given logits of (entity) from model, it will generate class labels and scores

        Args:
            entity_hs ([torch.tensor]): [num_sentence x seq_len x classes]
            word_pieces ([Array]): [Sentence word peices from bert tokenizers]
            tokenized_ids ([Array]): [Token ids from bert tokenizers pre-padding stage]

        Returns:
            words_labels [dict]: [Dictionary of shape (words_pieces x prediction_labels) eg. {'wake':'O', '5':'time'...}]
            words_scores  [dict]: [Dictionary of shape (words_pieces x num_classes) eg. {'wake':[class1:score,......], '5':[class1:score,.....]...}]
        """
        entity_scores,entity_preds = self.to_yhat(entity_hs)
        entity_scores = entity_scores[1:len(tokenized_ids)-1, :]
        enitity_labels = self.enc_entity.inverse_transform(entity_preds)[1:len(tokenized_ids)-1]
        words_labels_json = self.words_to_labels_json( word_pieces , enitity_labels)
        words_scores_json = self.words_to_scores_json( word_pieces, entity_scores)
        return words_labels_json,words_scores_json
        
    def classification(self,logits,task='intent'):
        """Given logits of (intent or scenario) from model, it will generate class labels and scores
        Args:
            logits ([torch.tensor]): [Tensor of shapee (Num_sentences x classes)]
            task (str, optional): [The classification task to perform]. Defaults to 'intent'.

        Returns:
            sentence_labels_json [dict]:  [Dictionary of shape (Num_sentences,) eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
            class_scores_json [dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}
        """
        if task == 'intent':
            enc = self.enc_intent
        else:
            enc = self.enc_scenario
        class_scores,class_preds = self.to_yhat(logits)
        sentence_labels_json = self.sentence_to_labels_json(enc.inverse_transform(class_preds), len(class_preds))
        class_scores_json = self.classes_to_scores_json(enc.classes_,class_scores)
        return sentence_labels_json, class_scores_json
    
    def predict(self,sentence):
        """Given a sentence it will return NLU model prediction for entity,intent and scenario, wrapped as  
        json object.
        Args:
            sentence ([str]): [Input string]
        Returns:
            words_labels_json [dict]: [Dictionary of shape (words_pieces x prediction_labels) eg. {'wake':'O', '5':'time'...}]
            words_scores_json  [dict]: [Dictionary of shape (words_pieces x num_classes) eg. {'wake':[class1:score,......], '5':[class1:score,.....]...}]
            intent_sentence_labels_json [dict]: [Dictionary of shape (Num_sentences,), eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
            intent_class_scores_json [dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}]
            scenario_sentence_labels_json [dict]:  [Dictionary of shape (Num_sentences,) eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
            scenario_class_scores_json [dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}
        """
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

nlu_engine = None
@app.route("/test")
def predict(verbose=False):
    print(request.args)
    sentence = request.args.get('sentence')
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
    parser = argparse.ArgumentParser(description="Demo for Natural language understanding")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use optimize_graph.py')
    args = parser.parse_args()
    

    nlu_engine = NLUEngine(args.model_file)

    
    app.run(debug=True)