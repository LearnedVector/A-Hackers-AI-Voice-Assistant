
from nltk import tokenize
from numpy.core.fromnumeric import sort
import pandas
from pandas.core.reshape.merge import merge


if __name__ == "__main__":
    # %%
    import pandas as pd
    import config
    df = pd.read_csv(config.DATASET_PATH,sep=';', encoding='latin-1')    
    df.tail()
    
    # %%
    df = df.filter(items=['scenario','intent','answer_annotation','answer_normalised'], axis=1)
    df['Sentence #'] = df.index
    df = df[df.answer_annotation.notnull() | df.answer_normalised.notnull()]
    df['answer_normalised'] =df['answer_normalised'].fillna(df['answer_annotation'] ) 
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
    df['answer_normalised'] = df['answer_normalised'].apply(process_sentence)
    # %%
    import re
    def extract_entity(s, i_e_dict):
        s = " ".join(s.split())
        if ':' in s:
            # print(s)
            s = re.split(':',s) #[time, five am]
            words = s[1].split() #[five, am]
            i_e_dict['entity'] += s[0:1] * len(words)
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
    
    df['entity']= df['answer_annotation'].apply(get_entity)
    df['word']= df['answer_annotation'].apply(get_words)
    # %%

    # %%
    df_ir = df.filter(items=['Sentence #','scenario','intent'], axis=1)
    # %%
    for i,s in  enumerate(df['answer_normalised'].values):
        if '[' in s:
            print(i,s)

    # %%
    df.answer_annotation.notnull() |  df.answer_normalised.notnull() 

    # %%

    # %%
     # %%

# %%
