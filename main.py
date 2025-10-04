# imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


data = pd.read_csv("spam.csv", encoding='latin1')
#print(data.head())

# remove unwanted columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
#print(data.head())

# check for null values
print(data.isnull().sum())
# --> returns none missing



# function to preprocess all of the text
def preprocess_text(text):
    # remove all URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # remove all whitespace
    text = re.sub('\s+', ' ', text).strip()
    # lower message
    text = text.lower()
    return text

data['cleaned_message'] = data['text'].apply(preprocess_text)
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(data['cleaned_message'])
data['label'] = np.where(data["label"] == 'spam', 1, 0)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#print(data.head())

model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def evaluate_model(model, X_test, y_test):
     prediction = model.predict(X_test)
     accuracy = accuracy_score(y_test, prediction)
     matrix = confusion_matrix(y_test, prediction)
     return accuracy, matrix

accuracy, matrix = evaluate_model(model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
















