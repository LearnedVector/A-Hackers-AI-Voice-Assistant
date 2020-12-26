import transformers
import torch
RAW_DATASET_PATH = '../data/nlu.csv'
ER_DATASET_PATH  = '../data/er_dataset.csv'
IS_DATASET_PATH  = '../data/is_dataset.csv'
MODEL_PATH = '../models/best_model.pth'
LOG_PATH = '../log'

#Additional Parameters:
SAVE_MODEL = False
#Hyper-parameters
MAX_LEN = 61+2
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
EPOCHS = 100

#Model selection parameter
BASE_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL,
    do_lower_case = True
)

#device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
