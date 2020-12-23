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

## Question
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
        
