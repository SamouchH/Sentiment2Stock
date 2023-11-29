import streamlit as st
import pandas as pd
#Other necessary imports
from transformers import BertTokenizer, BertForSequenceClassification
import torch

MODEL_PATH = './models_LLM/model.pth'

#Function to load the model
@st.cache_resource()
def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', #Use the 12-layer BERT model, with an uncased vocab
        num_labels=3, #The number of output labels
        output_attentions=False, #Whether the model returns attentions weights
        output_hidden_states=False #Whether the model returns all hidden-states
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_sentiment(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)
        return prediction.item()
    

def app():
    st.title('Sentiment Analysis with BERT')

    user_input = st.text_area("Enter text for sentiment analysis")

    if st.button('Analyze'):
        with st.spinner('Analyzing...'):
            prediction = predict_sentiment(user_input)
            sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}.get(prediction)
            st.write('Prediction: ', sentiment)

    
