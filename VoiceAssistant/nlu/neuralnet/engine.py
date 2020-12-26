from logging import FATAL
import torch.nn as nn
import torch

from model import NLUModel
def loss_func(logits, targets, mask, num_labels, entity=False):
    criterion = nn.CrossEntropyLoss()
    if entity:
        active_loss = mask.view(-1) == 1
        #generate new targets, such that the ceriterion ignore the targets for padding tokens
        active_targets = torch.where(
            active_loss,
            targets.view(-1),
            torch.tensor(criterion.ignore_index).type_as(targets)
        )
        
        logits = logits.view(-1,num_labels) # remove Tx-sample grouping

        loss = criterion(logits,active_targets)
    else:
        loss = criterion(logits,targets.view(-1))
    return loss


def eval_fn(data_loader,model,device,batch):
    model.eval()
    final_loss = 0
    with torch.no_grad():
        # for batch in data_loader:
        for k,v in batch.items():
            batch[k] = v.to(device)
        (entity_logits,
        intent_logits,
        scenario_logits) = model(batch['ids'], 
                                batch['mask'],
                                batch['token_type_ids'])

        entity_loss =  loss_func(entity_logits,batch['target_entity'],batch['mask'],model.num_entity, entity=True)
        intent_loss =  loss_func(intent_logits,batch['target_intent'],batch['mask'],model.num_intent)
        scenario_loss =  loss_func(scenario_logits,batch['target_scenario'],batch['mask'],model.num_scenario)
        
        loss = entity_loss + intent_loss + scenario_loss

        final_loss += loss
    return final_loss/len(data_loader)
def train_fn(data_loader,
             model,
             optimizer,
             scheduler,
             device,
             batch):

    model.train()
    final_loss = 0
    # for batch in data_loader:
    for k,v in batch.items():
        batch[k] = v.to(device)

    optimizer.zero_grad()

    (entity_logits,
        intent_logits,
        scenario_logits) = model(batch['ids'], 
                                batch['mask'],
                                batch['token_type_ids'])

    entity_loss =  loss_func(entity_logits,batch['target_entity'],batch['mask'],model.num_entity, entity=True)
    intent_loss =  loss_func(intent_logits,batch['target_intent'],batch['mask'],model.num_intent)
    scenario_loss =  loss_func(scenario_logits,batch['target_scenario'],batch['mask'],model.num_scenario)

    loss = entity_loss + intent_loss + scenario_loss
    loss.backward()

    optimizer.step()
    scheduler.step()

    final_loss += loss
    return final_loss/len(data_loader)
