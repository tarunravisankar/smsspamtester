import streamlit as st
import joblib
import re


model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  
    text = re.sub(r'[^\w\s]', '', text)                  
    text = re.sub('\s+', ' ', text).strip()             
    text = text.lower()                                  
    return text

st.title("SMS Spam Detector")
st.write("Enter an SMS message below and click 'Predict' to see if it's Spam or Ham.")

message = st.text_area("Type your message here:")

if st.button("Predict"):
    if message:
        cleaned_message = preprocess_text(message)
        vect_message = vectorizer.transform([cleaned_message])
        prediction = model.predict(vect_message)[0]
        prediction_label = "Spam" if prediction == 1 else "Ham"
        st.success(f"Prediction: {prediction_label}")
    else:
        st.warning("Please enter a message to predict!")

