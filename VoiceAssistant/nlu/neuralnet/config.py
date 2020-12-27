import transformers
import torch
RAW_DATASET_PATH = '../data/nlu.csv'
ER_DATASET_PATH  = '/home2/ncwn67/A-Hackers-AI-Voice-Assistant/VoiceAssistant/nlu/data/er_dataset.csv'
IS_DATASET_PATH  = '/home2/ncwn67/A-Hackers-AI-Voice-Assistant/VoiceAssistant/nlu/data/is_dataset.csv'
MODEL_PATH = '/home2/ncwn67/models/best_model.pth'
LOG_PATH = '/home2/ncwn67/A-Hackers-AI-Voice-Assistant/VoiceAssistant/nlu/logs/run2'

#Additional Parameters:
SAVE_MODEL = False
#Hyper-parameters
MAX_LEN = 61+2
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
EPOCHS = 10

#Model selection parameter
BASE_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL,
    do_lower_case = True
)

#device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
