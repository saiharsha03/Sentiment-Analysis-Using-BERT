import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embeddings(sentence):
        inputs = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings

with open("logistic_regression_model.pkl", 'rb') as file:
    clf = pickle.load(file)

st.title("Sentiment Analysis")
input_text = st.text_input("Enter your sentence here")

if input_text:
    result = clf.predict(get_sentence_embeddings(input_text))
    if result == 1:
        st.write(
            "<div style='text-align: center; background-color: green; font-size: 24px; color: white;'>Positive Review 😀</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align: center; background-color: red; font-size: 24px; color: white;'>Negative Review😔</div>",
            unsafe_allow_html=True
        )