# %%
from torch.utils.data import DataLoader
import pandas as pd 
import joblib
from sklearn import preprocessing
from sklearn import model_selection

import config 
from dataset import NLUDataset 
from model import NLUModel

n = NLUModel(2,2,2)
for n,p in n.named_parameters():
    print(n)
# %%
def process_entity_data(data_path):
    """Loads the preprocessed entity recognition dataset and returns the 
    X and Y needed for fitting bert to do entity recognition.
    Args:
        data_path ([str]): [Path to the preprocess entity-recognition dataset]

    Returns:
        sentences [Array]: [Array of shape (Samples,)]
                           eg. [["This", ",", "world", "is", "needs", "AI"], ["hello".....]]
        entity [Array]: [Array of shape (Samples, )]
                        eg.[[1, 2,3, 4,1,5], [....].....]]
        enc_entity: [LabelEncoder object for storing metadata]
        len_upper [int]: [Length of the longest sentence]
                         For the er dataset it should be 61
        
    """
    df = pd.read_csv(data_path ,encoding='latin-1')
    
    enc_entity = preprocessing.LabelEncoder()

    df.loc[:, 'entity'] = enc_entity.fit_transform(df['entity'])

    sentences = df.groupby('Sentence #')['words'].apply(list).values
    entity =  df.groupby('Sentence #')['entity'].apply(list).values
    len_upper = len(max(sentences, key=len))
    return sentences, entity , enc_entity,len_upper

def proccess_itent_scenario_data(data_path):
    """Loads the preprocessed itent and scenario classification dataset and returns the 
    X and Y needed for fitting bert to do entity classfication.
    Args:
        data_path ([str]): [Path to the preprocess intent-scenario-classfication dataset]

    Returns:
        intent [Array]: [Array of shape (Samples, ) containing intent class ids for each samples]
        scenario [Array]: [Array of shape (Samples, ) containing scenario class ids for each samples]
        enc_intent: [LabelEncoder object for storing metadata]
        enc_scenario: [LabelEncoder object for storing metadata]
    """
    df = pd.read_csv(data_path, encoding='latin-1')

    enc_intent = preprocessing.LabelEncoder()
    df.loc[:,'intent'] = enc_intent.fit_transform(df['intent'])

    enc_scenario = preprocessing.LabelEncoder()
    df.loc[:,'scenario'] = enc_scenario.fit_transform(df['scenario'])

    
    intent = df.groupby('Sentence #')['intent'].apply(list).values
    scenario = df.groupby('Sentence #')['scenario'].apply(list).values

    return intent,scenario,enc_intent, enc_scenario

def run():
    sentences, target_entity , enc_entity = process_entity_data(config.ER_DATASET_PATH)
    target_intent,target_scenario,enc_intent, enc_scenario = proccess_itent_scenario_data(config.IS_DATASET_PATH)
    
    num_entity, num_intent, num_scenario = len(enc_entity.classes_),len(enc_intent.classes_),len(enc_scenario.classes_)
    meta_data = {
        'enc_entity': enc_entity,
        'enc_intent': enc_intent,
        'enc_scenario': enc_scenario
    }
    joblib.dump(meta_data, 'meta_data.bin')
    
    (train_sentences,
     test_sentences,
     train_entity,
     test_entity,
     train_intent,
     test_intent,
     train_scenario,
     test_scenario
     )= model_selection.train_test_split(
            sentences,
            target_entity,
            target_intent,
            target_scenario,
            random_state=50,
            test_size=0.1
        )
    train_dataset = NLUDataset(train_sentences,
                               train_entity,
                               train_intent,
                               train_scenario)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size = config.TRAIN_BATCH_SIZE,
                                   num_workers= 4)
    test_dataset = NLUDataset(test_sentences,
                              test_entity,
                              test_intent,
                              test_scenario)
    test_data_loader = DataLoader(test_dataset,
                                   batch_size = config.TEST_BATCH_SIZE,
                                   num_workers=1)

    device = config.DEVICE 
    net = NLUModel(num_entity, num_intent, num_scenario)
    net.to(device)

    param_optimizer = list(net.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {
            'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_deacy': 0.001
        },
        {
            'params': [p for n,p in param_optimizer if  any(nd in n for nd in no_decay)],
            'weight_deacy': 0.0
        }
        
    ]
    
    
    
    

    
    
    
    

# %%
if __name__ == "__main__":

    # %%
    s, e , enc = process_entity_data(config.ER_DATASET_PATH)
    # %%
    # %%
    i,s,ei,es = proccess_itent_scenario_data(config.IS_DATASET_PATH)






# %%
