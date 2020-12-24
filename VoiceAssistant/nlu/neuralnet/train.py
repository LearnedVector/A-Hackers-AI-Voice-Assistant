# %%
import pandas as pd 
import config 
from sklearn import preprocessing


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


    

# %%
if __name__ == "__main__":

    # %%
    s, e , enc = process_entity_data(config.ER_DATASET_PATH)
    # %%
    max_
    # %%
    i,s,ei,es = proccess_itent_scenario_data(config.IS_DATASET_PATH)






# %%
