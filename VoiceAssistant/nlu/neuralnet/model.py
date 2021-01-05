import torch.nn as nn
import transformers
import config 
class NLUModel(nn.Module):
    def __init__(self,num_entity, num_intent, num_scenario):
        super(NLUModel,self).__init__()
        self.num_entity = num_entity
        self.num_intent = num_intent
        self.num_scenario = num_scenario

        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL
        ) 
        self.drop_1 = nn.Dropout(0.3)
        self.drop_2 = nn.Dropout(0.3)
        self.drop_3 = nn.Dropout(0.3)

        self.out_entity = nn.Linear(768,self.num_entity)
        self.out_intent = nn.Linear(768,self.num_intent)
        self.out_scenario = nn.Linear(768,self.num_scenario)

    def forward(self, ids,mask,token_type_ids):
        out = self.bert(input_ids=ids,
                              attention_mask=mask,
                              token_type_ids=token_type_ids
                              )
        hs, cls_hs = out['last_hidden_state'], out['pooler_output']
        entity_hs = self.drop_1(hs)
        intent_hs = self.drop_2(cls_hs)
        scenario_hs = self.drop_3(cls_hs)

        entity_hs = self.out_entity(entity_hs)
        intent_hs = self.out_intent(intent_hs)
        scenario_hs = self.out_scenario(scenario_hs)

        return entity_hs,intent_hs,scenario_hs

        


