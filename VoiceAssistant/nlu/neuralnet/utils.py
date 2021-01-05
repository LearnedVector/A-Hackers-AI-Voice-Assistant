"""Contains helper functions for train.py used for train and evaluating model """
import torch.nn as nn
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from model import NLUModel
from tqdm import tqdm


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


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    if 'Entity' in title:
        figure = plt.figure(figsize=(40, 40))
    elif 'Intent' in title:
        figure = plt.figure(figsize=(35, 35))
    else:
        figure = plt.figure(figsize=(13, 13))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
        
    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def get_confusion_matrix(y_hat,targets,enc):
    y_hat = enc.inverse_transform(y_hat)
    targets = enc.inverse_transform(targets)
    class_names = enc.classes_

    cm = confusion_matrix(y_hat, targets,labels=class_names)
    if len(class_names) == 57:
        title = 'Entity Confusion Matrix'
    elif len(class_names) == 54:
        title = 'Intent Confusion Matrix'
    else:
        title = 'Scenario Confusion Matrix'

    fig = plot_confusion_matrix(cm,class_names,title)
    return fig

    
def get_precision_recall(y_hat,targets):
    targets = targets
    percision, recall , fs, _ = precision_recall_fscore_support(y_hat, targets, average='micro')
    return percision,recall,fs 

def classifcation_report(y_hat,targets,enc):
    #function to get precsion_recall after each epoch
    percision,recall,fs = get_precision_recall(y_hat.flatten(),targets.flatten())
    fig = get_confusion_matrix(y_hat.flatten(),targets.flatten(),enc)
    return percision,recall,fs,fig

def to_yhat(logits):
    logits = logits.view(-1, logits.shape[-1])
    probs = torch.softmax(logits, dim=1)
    y_hat = torch.argmax(probs, dim=1).cpu().detach().numpy()
    return y_hat 

def test_fn(data_loader,model,device,enc_list):
    '''
    This function evalutes the test set and returns evaluation metrics 
    '''
    model.eval()
    final_loss = 0
    # tasks_y_hats = [None, None, None]
    # tasks_targets = [None, None, None]
    tasks_y_hats = [ None, None]
    tasks_targets = [ None, None]

    precision_dict,recall_dict,fs_dict,fig_dict = {},{},{},{}
    with torch.no_grad():
        for bi, batch in enumerate(data_loader):
            for k,v in batch.items():
                batch[k] = v.to(device)
            # (entity_logits,
            # intent_logits,
            # scenario_logits) = model(batch['ids'], 
            #                         batch['mask'],
            #                         batch['token_type_ids'])
            
            (intent_logits,
            scenario_logits) = model(batch['ids'], 
                                    batch['mask'],
                                    batch['token_type_ids'])
            
            # entity_loss =  loss_func(entity_logits,batch['target_entity'],batch['mask'],model.num_entity, entity=True)
            intent_loss =  loss_func(intent_logits,batch['target_intent'],batch['mask'],model.num_intent)
            scenario_loss =  loss_func(scenario_logits,batch['target_scenario'],batch['mask'],model.num_scenario)
            
            # loss = (entity_loss + intent_loss + scenario_loss)/3
            loss = (intent_loss + scenario_loss)/2
            final_loss += loss
            
            # targets_keys = ['target_entity', 'target_intent','target_scenario']
            # logits_list = [entity_logits, intent_logits, scenario_logits]
            targets_keys = ['target_intent','target_scenario']
            logits_list = [ intent_logits, scenario_logits]
            
            for i,(y_hats,target,target_key,logits) in enumerate(zip(tasks_y_hats,tasks_targets,targets_keys,logits_list)):
                if not (y_hats is None):
                    tasks_y_hats[i] = np.concatenate((y_hats,to_yhat(logits)))
                    tasks_targets[i] = np.concatenate((target,batch[target_key].cpu()))
                    
                    #also need to stack the targets
                else:
                    tasks_y_hats[i] = to_yhat(logits)
                    tasks_targets[i] = batch[target_key].cpu()

       
        #the code below should be done after all batches proccess
        # for y_hats,target,enc,key in zip(tasks_y_hats,
        #                                         tasks_targets,
        #                                         enc_list,
        #                                         ['entity', 'intent' , 'scenario']):
        #     precision,recall,fs,fig = classifcation_report(y_hats, target
        #                                                    ,enc)

        #     precision_dict[key] = precision
        #     recall_dict[key] = recall
        #     fs_dict[key] = fs
        #     fig_dict[key] = fig
        for y_hats,target,enc,key in zip(tasks_y_hats,
                                                tasks_targets,
                                                enc_list,
                                                [ 'intent' , 'scenario']):
            precision,recall,fs,fig = classifcation_report(y_hats, target
                                                           ,enc)

            precision_dict[key] = precision
            recall_dict[key] = recall
            fs_dict[key] = fs
            fig_dict[key] = fig
    return final_loss/len(data_loader), precision_dict,recall_dict,fs_dict,fig_dict

def eval_fn(data_loader,model,device,n_examples,batch=None):
    model = model.eval()
    total_words = 0
    correct_predictions_entity = 0
    correct_predictions_intent = 0
    correct_predictions_scenario = 0
    final_loss = 0
    with torch.no_grad():
        for bi,batch in enumerate(data_loader):
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
            
            loss = (entity_loss +intent_loss + scenario_loss)/3
            final_loss += loss

            total_words += (entity_logits.shape[0] * entity_logits.shape[1])
            _, preds_entity = torch.max(entity_logits,dim=-1)
            _,preds_intent = torch.max(intent_logits, dim=1)
            _, preds_scenario = torch.max(scenario_logits,dim=1)

            correct_predictions_entity += torch.sum(preds_entity.view(-1) == batch['target_entity'].view(-1))
            correct_predictions_intent += torch.sum(preds_intent == batch['target_intent'].view(-1))
            correct_predictions_scenario += torch.sum(preds_scenario == batch['target_scenario'].view(-1))
    return final_loss/len(data_loader) ,correct_predictions_entity.double()/total_words, correct_predictions_intent.double()/n_examples,  correct_predictions_scenario.double()/n_examples 
def train_fn(data_loader,
             model,
             optimizer,
             scheduler,
             device,
             n_examples,
             batch = None
             ):

    model = model.train()
    final_loss = 0
    correct_predictions_entity = 0
    correct_predictions_intent = 0
    correct_predictions_scenario = 0
    total_words = 0
    for bi, batch in enumerate(data_loader):
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

        loss = (entity_loss + intent_loss + scenario_loss)/3
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_words += (entity_logits.shape[0] * entity_logits.shape[1])
        _, preds_entity = torch.max(entity_logits,dim=-1)
        _,preds_intent = torch.max(intent_logits, dim=-1)
        _, preds_scenario = torch.max(scenario_logits,dim=-1)

        correct_predictions_entity += torch.sum(preds_entity.view(-1) == batch['target_entity'].view(-1))
        correct_predictions_intent += torch.sum(preds_intent == batch['target_intent'].view(-1))
        correct_predictions_scenario += torch.sum(preds_scenario == batch['target_scenario'].view(-1))
        final_loss += loss.item()
    return final_loss/len(data_loader),correct_predictions_entity.double()/total_words , correct_predictions_intent.double()/n_examples,  correct_predictions_scenario.double()/n_examples 
