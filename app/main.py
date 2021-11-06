import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalAveragePooling2D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import Model, load_model
from keras import regularizers
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

stop_words = set(stopwords.words('english'))
PAGE_CONFIG = {'page_title':'IRE Major Project','layout':"wide"}
st.set_page_config(**PAGE_CONFIG)


# cleaning data
def clean_post(post):
    post = post.lower()
    post = re.sub(r"\n", " ", post)
    post = re.sub("[\<\[].*?[\>\]]", " ", post)
    post = re.sub(r"[^a-z ]", " ", post)
    post = re.sub(r"\b\w{1,3}\b", " ", post)
    return " ".join([x for x in post.split() if x not in stop_words])    

def predict_text_label_CNN(text):
    X_test = np.array([clean_post(text),])
    # Convert  texts to sequence of integers
    sequences_test = st.session_state.tokenizer.texts_to_sequences(X_test)
    # Limit size of test sequences to 200 and pad the sequence
    X_test = pad_sequences(sequences_test, maxlen=200)
    # Evaluating
    pred_test = st.session_state.cnn_model.predict(X_test)
    return np.argmax(pred_test, axis=1)[0]

# Saved models are loaded here
def load_all_models():
    # Load tokenizer
    with open('../models/tokenizer.pkl', 'rb') as f:
        st.session_state.tokenizer = pickle.load(f)
    # CNN
    st.session_state.cnn_model = load_model(f'../models/CNN_model')


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

