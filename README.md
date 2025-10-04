# SMS Spam Predictor

[![Open in Streamlit](https://img.shields.io/badge/Open_in-Streamlit-brightgreen?logo=streamlit)](https://smsspamtester-hv8l26waslftt5mjnfe9x3.streamlit.app/)

A simple machine learning project that predicts whether a given SMS message is **spam** or **ham (not spam)**. This app uses a trained **Naive Bayes classifier** and a **TF-IDF vectorizer** to analyze SMS messages.

## Demo

Check out the live demo here: [SMS Spam Predictor](https://smsspamtester-hv8l26waslftt5mjnfe9x3.streamlit.app/)

## Features

- Predicts spam or ham messages in real-time.
- User-friendly Streamlit interface.
- Lightweight and fast ML model.
- Easy to deploy and customize.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-predictor.git

2. Navigate to the project directory:
  cd sms-spam-predictor

3. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

5. Install dependencies:
pip install -r requirements.txt

6. Run the Streamlit app:
streamlit run app.py

Model:
Algorithm: Multinomial Naive Bayes
Vectorizer: TF-IDF=
Dataset: Public SMS Spam Collection Dataset
