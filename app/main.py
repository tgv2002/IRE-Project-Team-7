import torch
import pandas as pd
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler,RandomSampler
from transformers import DistilBertForSequenceClassification, AdamW, BertConfig
# from transformers import get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import os
import re
import shutil
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import streamlit as st

def model_init_():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    tokenizer = DistilBertTokenizer.from_pretrained('model_save_epoch3/', do_lower_case=True)
    num_labels = 15
    batch_size = 32
    model = DistilBertForSequenceClassification.from_pretrained(
        "model_save_epoch3/", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = num_labels, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    torch.cuda.empty_cache()
    model.cuda()
    return device,tokenizer,num_labels,batch_size,model

# cleaning data
def clean_post(post):
    post = post.lower()
    post = re.sub(r"\n", " ", post)
    post = re.sub("[\<\[].*?[\>\]]", "", post)
    # post = re.sub(r"[^a-zA-Z ]", "", post)
    # post = re.sub(r"\b\w{1,3}\b", " ", post)
    return post

def get_metrics(y_true, y_pred,epoch):
    f = open(f'report{epoch}.txt','w+')
    result1 = classification_report(y_true, y_pred)
    print('Classification Report: ', result1)
    f.write('\nClassification Report: \n')
    f.write(result1)
    # print(type(result1))
    # df = pd.DataFrame(result1).transpose()
    # df.to_csv('report.csv')
    result2 = accuracy_score(y_true, y_pred)
    print('Accuracy: ', result2, "\n\n")
    f.write('\nAccuracy Report: \n')
    f.write(str(result2))
    
def create_dataset(tokenizer,batch_size,data):
    X = data['post'].apply(lambda post: clean_post(post))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(np.array(data['mental_disorder']))
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for ind,sent in enumerate(X):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 256,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True, # Construct attn. masks.
                            truncation=True, 
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(y,dtype=torch.long)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', X.iloc[0])
    print('Token IDs:', input_ids[0])
    from torch.utils.data import TensorDataset, random_split

    # Combine the training inputs into a TensorDataset.
    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return test_dataloader

def eval(model,test_dataloader,device):
    model.eval()
    predictions ,predictions_final, true_labels = [], [], []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions_final += np.argmax(logits, axis=1).flatten().tolist()
        predictions += logits.tolist()
        true_labels += label_ids.tolist()
  
    return predictions_final,true_labels

def evaluate_on_test_data():
    device,tokenizer,num_labels,batch_size,model = model_init_()
    data = pd.read_csv('reddit_dataset/test.csv')[['post','mental_disorder']]
    data = shuffle(data)
    print(len(data))
    test_dataloader = create_dataset(tokenizer,batch_size,data)
    predictions_final, true_labels = eval(model,test_dataloader,device)
    get_metrics(true_labels,predictions_final,"test")
    
def get_text_label(text):
    device,tokenizer,num_labels,batch_size,model = model_init_()
    data = pd.DataFrame({
        'post':[text],
        'mental_disorder': ["None"]
    })
    test_dataloader = create_dataset(tokenizer,batch_size,data)
    predictions_final, true_labels = eval(model,test_dataloader,device)
    return predictions_final[0]

PAGE_CONFIG = {'page_title':'IRE Major Project','layout':"wide"}

st.set_page_config(**PAGE_CONFIG)
 
if __name__ == '__main__':
    
    label_ids_to_names = {
    0: "Eating disorder", 1: "Addiction", 2: "ADHD", 3: "Alcoholism",
    4: "Anxiety", 5: "Autism", 6: "Bi-polar disorder", 7: "Borderline personality disorder", 
    8: "Depression", 9: "Health Anxiety", 10: "Loneliness", 11: "PTSD", 
    12: "Schizophrenia", 13: "Social Anxiety", 14: "Suicidal tendencies"
    }
    
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = True
        load_all_models()
        
    st.title("Multi-Class Classification of Mental Health Disorders")
    st.write('\n'*12)

    entered_text = st.text_area(label='Enter required text in the text area below, which would be used to label it with its most appropriate mental disorder')

    col1, col2, col3 , col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        submit = st.button('Submit')

    if submit:
        # Model is used for predicting label for the text here
        predicted_label = predict_text_label_CNN(entered_text)
        st.markdown(f"### Predicted disorder: {label_ids_to_names[predicted_label]}")
        

