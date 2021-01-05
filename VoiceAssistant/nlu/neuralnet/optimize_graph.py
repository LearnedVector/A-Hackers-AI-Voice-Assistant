import torch
from model import NLUModel
import config

DEVICE = config.DEVICE

def trace(model):
    model.eval()
    ids = torch.zeros(1,61, dtype=torch.long)
    cls = torch.tensor([[101]],dtype=torch.long)
    sep = torch.tensor([[102]],dtype=torch.long)
    ids = torch.cat((cls,ids,sep),dim=1)
    
    mask = torch.zeros(1,63,dtype=torch.long)
    token_type_ids = torch.zeros(1,63,dtype=torch.long)
    traced = torch.jit.trace(model,ids,mask,token_type_ids)
    return traced
def main():
    MODEL = NLUModel(57,54,18)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH,
                                    map_location= lambda storage, loc:storage))
    MODEL.to(DEVICE)

    print('tracing model')
    traced_model = trace(MODEL)
    print('Saving traced model to ', config.TRACE_MODEL_PATH)
    traced_model.save(config.TRACE_MODEL_PATH)
    print('Done!')

if __name__ == "__main__":
    main()   


