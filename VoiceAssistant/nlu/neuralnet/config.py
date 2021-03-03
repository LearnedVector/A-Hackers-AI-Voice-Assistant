import transformers
import torch
RAW_DATASET_PATH = '../nlu_dataset/nlu.csv'
ER_DATASET_PATH  = '../nlu_dataset/er_dataset.csv'
IS_DATASET_PATH  = '../nlu_dataset/is_dataset.csv'
MODEL_PATH = '/Volumes/My Passport 1/models/epoch50_best_model.pth' 
TRACE_MODEL_PATH ='/Volumes/My Passport 1/models/epoch50_best_model_trace.pth' 
LOG_PATH = '/home2/ncwn67/A-Hackers-AI-Voice-Assistant/VoiceAssistant/nlu/tb_logs/run4'

#Additional Parameters:
SAVE_MODEL = True
#Hyper-parameters
MAX_LEN = 61+2
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
EPOCHS = 50

#Model selection parameter
BASE_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL,
    do_lower_case = True
)

#device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
