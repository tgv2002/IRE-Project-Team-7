import streamlit as st
import random

PAGE_CONFIG = {'page_title':'IRE Major Project','layout':"wide"}
st.set_page_config(**PAGE_CONFIG)

# Saved model is loaded here

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
    st.markdown(f"### Predicted disorder: {random.choice(['ADHD', 'BPD', 'Anxiety'])}")     # dummy output

