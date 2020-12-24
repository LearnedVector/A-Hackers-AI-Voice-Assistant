import transformers
RAW_DATASET_PATH = '../data/nlu.csv'
ER_DATASET_PATH  = '../data/er_dataset.csv'
IS_DATASET_PATH  = '../data/is_dataset.csv'

#Hyper-parameters
MAX_LEN = 61+2

#Model selection parameter
BASE_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL,
    do_lower_case = True
)
