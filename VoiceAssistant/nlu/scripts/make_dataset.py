#%%
import pandas as pd
import re
import sys
sys.path.append('../')
import neuralnet.config as config
'''
This is a script to  preprocess the Natural Language Understanding dataset introduced by Xingkun Liu in the paper
Benchmarking Natural Language Understanding Services for building Conversational Agents.
@InProceedings{XLiu.etal:IWSDS2019,
  author    = {Xingkun Liu, Arash Eshghi, Pawel Swietojanski and Verena Rieser},
  title     = {Benchmarking Natural Language Understanding Services for building Conversational Agents},
  booktitle = {Proceedings of the Tenth International Workshop on Spoken Dialogue Systems Technology (IWSDS)},
  month     = {April},
  year      = {2019},
  address   = {Ortigia, Siracusa (SR), Italy},
  publisher = {Springer},
  pages     = {xxx--xxx},
  url       = {http://www.xx.xx/xx/}
}
https://github.com/xliuhw/NLU-Evaluation-Data/blob/master/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv
An error was found  in a line of  the original csv dataset file and was corrected to the following:
980;25817;weather;query;IRR_XL;is the [place_name : city] hot or cold in terms of [weather_descriptor : temperature] [query_detail : if it is a coastal area] then is the [weather_descriptor : humidity] high or low;null;date, location, time;is the city hot or cold in terms of temperature if it is a coastal areathen is the humidity high or low;Is the city  hot or cold in terms of temperature .If it is a coastal area,then is the humidity high or low.;How would you ask your PDA about the weather in another city?
To utilize this script please rectify the error.
'''
def entity_recognition_dataset(df):
    df_er = df.filter(items=['Sentence #','word','entity'], axis=1)
    words = df_er['word'].apply(pd.Series)
    entity = df_er['entity'].apply(pd.Series)
    words = words.stack()
    entity = entity.stack()
    merge = pd.concat([words,entity], axis=1).reset_index()
    merge = merge.drop(labels='level_1' , axis = 1)
    merge.rename({'level_0' :'Sentence #' , '0':'words' , '1':'entity'})
    merge.columns = ['Sentence #' , 'words' , 'entity']
    return merge
def intent_scenario_classfication_dataset(df):
    merge = df.filter(items=['Sentence #','scenario','intent'], axis=1)
    return merge 

def process_sentence(d):
    s  = ''
    found = False
    for l in d:
        if l == '[': 
            found = True
        if not found and (l != ']'): s += l
        if l == ':': found = False
    s = " ".join(s.split())    
    return s 

def extract_entity(s, i_e_dict):
    s = " ".join(s.split())
    if ':' in s:
        # print(s)
        s = re.split(':',s) #[time, five am]
        words = s[1].split() #[five, am]
        entity = " ".join(s[0].split())
        i_e_dict['entity'] += [entity] * len(words)
    else:
        words = s.split() #[wake, me, up]
        i_e_dict['entity'] += ['O'] * len(words)
    return  i_e_dict
        
def extract_words(s, i_w_dict):
    s = " ".join(s.split())
    if ':' in s:
        # print(s)
        s = re.split(':',s) #[time, five am]
        words = s[1].split() #[five, am]
        i_w_dict['word'] += words
    else:
        words = s.split() #[wake, me, up]
        i_w_dict['word'] += words
    return i_w_dict
        
def get_words(s):
    # print(s)
    import re
    from collections import defaultdict
    i_w_dict  = defaultdict(list)
    s = re.split('\[(.*?)\]',s)
    for subsentence in s:
        if subsentence != ' ' and len(subsentence):
            i_w_dict = extract_words(subsentence, i_w_dict)
    return list(i_w_dict.values())[0]
def get_entity(s):
    # print(s)
    import re
    from collections import defaultdict
    i_e_dict  = defaultdict(list) 
    s = re.split('\[(.*?)\]',s)
    for subsentence in s:
        if subsentence != ' ' and len(subsentence):
            i_e_dict = extract_entity(subsentence,i_e_dict)
    return list(i_e_dict.values())[0]
def preprocess_raw_df(df):
    df = df.filter(items=['scenario','intent','answer_annotation','answer_normalised'], axis=1)
    df['Sentence #'] = df.index
    df = df[df.answer_annotation.notnull() | df.answer_normalised.notnull()]
    df['answer_normalised'] =df['answer_normalised'].fillna(df['answer_annotation'] ) 
    df['answer_normalised'] = df['answer_normalised'].apply(process_sentence)
    df['entity']= df['answer_annotation'].apply(get_entity)
    df['word']= df['answer_annotation'].apply(get_words)
    return df

def make_dataset():
    df = pd.read_csv(config.RAW_DATASET_PATH,sep=';', encoding='latin-1')    
    df = preprocess_raw_df(df)
    er_dataset = entity_recognition_dataset(df)
    is_dataset = intent_scenario_classfication_dataset(df)
    return er_dataset,is_dataset
#=== End of preprocessing ===#

#a,b = make_dataset()    
# %%
# a.to_csv('../data/er_dataset.csv')
# b.to_csv('../data/is_dataset.csv')
if __name__ == "__main__":
    print('Loading the dataset!')
    er_dataset ,is_dataset = make_dataset()
    print('Successfully loaded the dataset!')
