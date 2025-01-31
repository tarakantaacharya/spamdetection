import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model (ensure the path to your model is correct)
model = joblib.load('voting_classifier.pkl')

vectorizer = joblib.load('vectorizer.pkl')

# Title of the web app
st.title("SMS Spam Detection")

# Text input field for SMS
sms = st.text_area("Enter the SMS message:")

# Prediction logic
if sms:
    # Preprocess the input SMS (you can add preprocessing steps if needed)
    sms_vectorized = vectorizer.transform([sms])

    # Get the prediction
    prediction = model.predict(sms_vectorized)
    
    # Display the result
    if prediction[0] == 1:
        st.subheader("Prediction: This is a Spam message!")
    else:
        st.subheader("Prediction: This is a Ham message (not Spam)")