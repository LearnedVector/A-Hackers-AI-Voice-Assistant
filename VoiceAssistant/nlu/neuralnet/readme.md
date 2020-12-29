# Dataset 
    - Raw Coloumns
        - userid; [x]
        -answerid;[x]
        -scenario;
        -intent;
        -status; 
        -answer_annotation;[x]
        -notes;[x]
        -suggested_entities;[x]
        -answer_normalised;[x]
        -answer;[x]
        -question[x] 
    - Needed Coloums
        - answer_normalized 
            - this is the sentence 
        - asnwer_annotations
            - this contains the normalized sentence and the entity
        - scenario
        - intent

#### Question and To-do list for data-cleaning
    - What should the dataframe look like?
        - It best to have 2 data frame
            - df_er: sentence #, words, entity_tag
            - df_ir: sentence#, intent, scenario
                - we don't we intent for each word because, these are the targets for sentence classfication
    - I need a new coloum call words
        - Get it by spliting answer_normalized
        - replace nan with answer_annotation
    - Remove row if answer_annotation and answer_nomralized are both NaN[x]
    - Generate answer_normalized with out any brackets[x]
    - Generate entity tag for each word 
        - produce entity:text dict[x]
        - Produce a WordxEntity Table 
            - what are the entity for word with no actual entity?
                - 0 
            - Now given a sentence, can now produce word:entity tag
        - need strip the entity tag

### Questions and To-do list for creating model
    - Write a module/function to loaded the preproccess ouput and return required outputs [x]
        - What are the required outputs?
            - inputs for er model 
                - words grouped by sentence number
                - entities class ids, grouped by sentence number
            - inputs for the i_s classification model
                - words grouped by sentence number
                - itent class ids, grouped by senetence number
                - scenario class ids, grouped by senetence number
    - Tokenizing words
        - when tokenizing words using encode_plus, how do we adjust the label so that it takes into account word piece
            - We can't. Thus, using encode_plus is a bad idea here, for er task. why?
                - we need lables for each entity, with encode_plus we don't know how many sub-piece each word has.
                - We will have to iterate the words again taking 
                O(2n) complexity, instead O(n) 
        - Step
            - Encode the word
            - Update the label
            - Trim to seq
            - Add special tokens
                - ids 
                - label
            - Create attention mask
                - To distingush from padded tokens
            - Create token_type_ids
                - To indicate this we have sentence A in bert model
            - Do post padding for batching puropose
                - ids
                - label
    - Get confusion matrix and other metrics for test set 
         - I need to be able to store all y_hats for all the batches 
         - precsion,recall,cm, for each head
    - Labels
        - [1 x 63], we have lables for all 63 Tx
    - Y_hat
        - [1 x 57], there is only one Tx 
        - [1 x 63x 57] -> []       -> [1 x 63]
        - why is the logits of intent of shape [1 x 57], shouldn't it be?
            - [Sample x Classes ]
            - Because 
    - Why is the confusiont matx
### Notes
    - Increase the max_len by 2 to take into account the special token
    - Try weighted crossentropy loss becasue do to skewed distribution of classes 
    - Change the dataset such that non-entity has 0?
        - dont need, we need prediction for even non-entity words
    - Suggestion to refactor the code 
        - https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/7
    - Do I need multiple optimizers?
        - one way is to add the loss of all the classification task and use 1 optimizer to reach optimum
        - Do I need to send the parameters of bert to the optimizer?
            - No,
            - https://mccormickml.com/2019/07/22/BERT-fine-tuning/#4-train-our-classification-model
            - send all model paramters to the optimizer, including berts. Which makes sense, we need to adjust bert to our task. 
            - https://jg8610.github.io/Multi-Task/
            - Joint training is done when we have 1 dataset and multiple labels 
        - If yes, Do I need to separate out the parameters, so that entity optimizer don't get parameters of intent?
            - https://medium.com/@kajalgupta/multi-task-learning-with-deep-neural-networks-7544f8b7b4e3
            - https://github.com/google-research/bert/issues/504
    - Change engine so that it takes one optimizer, passing multiple optimizer is not scaleable and alot repeating code
    - Put starting lr to a hyperparameter


- Trying training the model on only individual task, 
    - Intent
        - Model does really bad on bs=16
        - What about bs=1?
            - It works, it learn to predict the class of the single sentence
        - Turns out the error was here 
            - num_train_steps = len(train_sentences) // (config.TRAIN_BATCH_SIZE * config.EPOCHS) 
            - scheduler =  get_linear_schedule_with_warmup(
                                                    optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=num_train_steps
                                                )
    - Scenairo 
    - Entity

Try training on only Cls task
    - Intent + Entity Joint
        - Now works

Try training on all 3 task
    - Now works

Generate test,validation and test set such that their distrubtion is same
    - That will mean we will need to validate data 3 times 
        - one with intent_valid_set
            - when we do this we will only care about intent_logits
        - one with scenario_valid_set
        - one with entity_valid_set
    - Split Train-test 3 times using stratify as intent,scenario,entity
    - Split Train-val 3 times using stratify as intent,scenario,entity

Fix validation

Get Results for 
    - Entity
        - produce the word_piece x label json (N x 1) [x]
        - Produce word_pieces x class_scores dict (N x C) [x]
    - Intent
        - produce label (1,) [x]
        - produce class_label x scores (N,C) [x]
    
    - Scenario
        - To the same [x]

Optimize the graph 
    - Save the file [x]
    - Test the traced model on app [x]

Rename the engine file to something else 
Rename app.py to engine.py



