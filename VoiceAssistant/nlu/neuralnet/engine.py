import torch.nn as nn
import torch
import tqdm
def loss_func(logits, targets, mask, num_labels):
    criterion = nn.CrossEntropyLoss()
    
    active_loss = mask.view(-1) == 1
    #generate new targets, such that the ceriterion ignore the targets for padding tokens
    active_targets = torch.where(
        active_loss,
        targets.view(-1),
        torch.tensor(criterion.ignore_index).type_as(targets)
    )
    
    logits = logits.view(-1,num_labels) # remove Tx-sample grouping

    loss = criterion(logits,active_targets)
    return loss

def train_fn(data_loader,
             model,
             entity_optimizer,intent_optimizer,scenario_optimizer,
             entity_scheduler,intent_scheduler,scenario_scheduler,
             device):
    model.train()
    final_entity_loss, final_intent_loss,final_scenario_loss = 0,0,0
    for batch in tqdm(data_loader, total=len(data_loader)):
        for k,v in batch.items():
            batch[k] = v.to(device)

        entity_optimizer.zero_grad()
        intent_optimizer.zero_grad()
        scenario_optimizer.zero_grad()

        (entity_logits,
         intent_logits,
         scenario_logits) = model(batch['ids'], 
                                  batch['mask'],
                                  batch['token_type_ids'])

        entity_loss =  loss_func(entity_logits,batch['target_entity'])
        entity_loss.backward()
        intent_loss =  loss_func(intent_logits,batch['target_intent'])
        intent_loss.backward()
        scenario_loss =  loss_func(scenario_logits,batch['target_scenario'])
        scenario_loss.backward()

        entity_optimizer.step()
        intent_optimizer.step()
        scenario_optimizer.step()

        entity_scheduler.step()
        intent_scheduler.step()
        scenario_scheduler.step()

        final_entity_loss += entity_loss
        final_intent_loss += intent_loss
        final_scenario_loss += scenario_loss
    return (final_entity_loss/len(data_loader),
            final_intent_loss/len(data_loader),
            final_scenario_loss/len(data_loader))