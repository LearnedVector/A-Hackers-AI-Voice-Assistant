# %%
from numpy.lib.function_base import select
import torch
import config



        
        
        
        
class IntentScenarioDataset(torch.utils.data.Dataset):
    def __init__(self,text,intent,scenario,
                 require_text=True):
        self.texts = text 
        self.intent = intent
        self.scenario = scenario

        self.require_text = require_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        text = self.texts[item]
        intent = self.intent[item]
        scenario = self.intent[item]

        ids, mask, token_type_ids = None, None, None
        if self.require_text:
            out = self.tokenizer.encode_plus(
                text,
                None,
                add_special_token=True,
                max_length = self.max_len,
                padding = 'max_length'
            )
            ids = out['input_ids']
            mask = out['attention_mask']
            token_type_ids = out['token_type_ids']
        return {
            'ids':ids,
            'target_intent':intent,
            'target_scenario':scenario,
            'mask': mask,
            'token_type_ids':token_type_ids
        }
            
class EntityDataset(torch.utils.data.Dataset):
    def __init__(self, text, entity):
        self.texts = text
        self.entity = entity
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN 
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        text = self.texts[item]
        entity = self.entity[item]

        ids = []
        target_entity = []
        for i,word in enumerate(text):
            token_ids = self.tokenizer.encode(word,
                                          add_special_tokens=False)
            word_piece_entity = [entity[i]]*len(token_ids)

            ids.extend(token_ids)
            target_entity.extend(word_piece_entity)

        #adujst to ids and target_entity to max_len, as max_len special_tokens inclusive
        ids = ids[:self.max_len-2]
        target_entity = target_entity[:self.max_len-2]
        #add cls token
        ids = [101] + ids + [102]
        target_entity = [0] + target_entity + [0]
        
        #create mask and token_type_id
        mask,token_type_id = [1]*len(ids),[0]*len(ids)

        #padding
        padding_len = self.max_len - len(ids)
        
        ids = ids + ([0] * padding_len)
        target_entity = target_entity + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_id = token_type_id + ([0] * padding_len)

        return {
            'ids': ids,
            'target_entity': target_entity,
            'mask': mask,
            'token_type_id': token_type_id
        }

class NLUDataset(torch.utils.data.Dataset):
    def __init__(self, text,entity,intent,scenario):
        self.texts = text
        self.entity = entity
        self.intent = intent
        self.scenario = scenario

        self.entity_dataset = EntityDataset(self.texts,
                                            self.entity)
        self.intent_scenario_dataset = IntentScenarioDataset(self.texts,
                                                             self.intent,
                                                             self.scenario,
                                                             require_text=False)
    def __len__(self):
        return len(self.texts)

    def __getitem__(self,item):
        entity_item = self.entity_dataset[item] 
        intent_scenario_item = self.intent_scenario_dataset[item]
        
        return {
            'ids':entity_item['ids'],
            'target_entity': entity_item['target_entity'],
            'target_intent': intent_scenario_item['target_intent'],
            'target_scenario':intent_scenario_item['target_scenario'],
            'mask': entity_item['mask'],
            'token_type_id': entity_item['token_type_id'],
        }
        
        
if __name__ == "__main__":
    test_text = [['hello','siri']]
    test_entity = [['3', '1']]
    test_intent, test_scenario = [[['3']],['2']]

    #test er dataset 
    expected_out = {'ids': [101, 7592, 2909, 2072, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'target_entity': [0, '3', '1', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'mask': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    er_dataset = EntityDataset(text=test_text,
                  entity=test_entity)
    out = er_dataset[0]
    assert out == expected_out

    #test i_s dataset 
    expected_out =  {'ids': [101, 7592, 100, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'target_intent': ['3'], 'target_scenario': ['3'], 'mask': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    is_dataset = IntentScenarioDataset(test_text,test_intent, test_entity)
    out = is_dataset[0]
    assert out == expected_out
    
    expected_out = {'ids': None, 'target_intent': ['3'], 'target_scenario': ['3'], 'mask': None, 'token_type_ids': None}
    is_dataset = IntentScenarioDataset(test_text,test_intent, test_entity, require_text=False)
    out = is_dataset[0]
    assert out == expected_out

    #test NLU wrapper dataset 
    expected_out = {'ids': [101, 7592, 2909, 2072, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'target_entity': [0, '3', '1', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'target_intent': ['3'], 'target_scenario': ['3'], 'mask': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    nlu_dataset = NLUDataset(test_text,
                  test_entity,
                  test_intent,
                  test_scenario
                  )
    out = nlu_dataset[0]
    assert out == expected_out



                
        

        
