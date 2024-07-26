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
    st.write(f"Prediction: {result}")