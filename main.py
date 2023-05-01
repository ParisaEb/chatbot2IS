import json
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
nltk.download('punkt')
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import os
from gtts import gTTS

with open('intentsComplete.json', encoding='utf-8') as file:
    data = json.load(file)
words = []
labels = []
docs_x = []
docs_y = []
for intent in data['data']:
    for pattern in intent['utterances']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs_x.append(pattern)
        docs_y.append(intent['intent'])

    if intent['intent'] not in labels:
        labels.append(intent['intent'])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
for intent in data['data']:
    for pattern in intent['utterances']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs_x.append(pattern)
        docs_y.append(intent['intent'])

    if intent['intent'] not in labels:
        labels.append(intent['intent'])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(docs_x)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = LinearSVC()
model.fit(X_train_tfidf, docs_y)

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)
def predict_intent(text, return_confidence=False):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    # Transform the text using the vectorizer and tf-idf transformer
    transformed_text = tfidf_transformer.transform(vectorizer.transform([processed_text]))
    # Predict the intent using the model
    confidence = None
    if return_confidence:
        predicted_proba = model._predict_proba_lr(transformed_text)[0]
        confidence = max(predicted_proba) * 100
    predicted_intent = model.predict(transformed_text)
    if return_confidence:
        return predicted_intent[0], confidence
    return predicted_intent[0]

responses = {}
if isinstance(data, dict) and 'data' in data:
    for d in data['data']:
        if isinstance(d, dict) and 'intent' in d and 'answers' in d:
            responses[d['intent']] = d['answers']

def respond(user_input):
    # Predict the intent using the model
    intent, confidence = predict_intent(user_input, return_confidence=True)

    # Check if the intent is above a certain confidence threshold
    if confidence > 1.5:
        if intent in responses:
            tag = intent
            response = random.choice(responses[tag])
            return response

    return "I'm sorry, I didn't understand your question."

while True:
    user_input = input("Ask a question about the Master program or say 'goodbye' to exit: ")
    if user_input.lower() == 'goodbye':
        print("Goodbye!")
        break
    s = respond(user_input)
    print(s)

import os

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Print the directory
print(script_dir)












