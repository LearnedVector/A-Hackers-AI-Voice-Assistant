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
            