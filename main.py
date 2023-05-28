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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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
        predicted_intent = model.predict(transformed_text)
        predicted_proba = model._predict_proba_lr(transformed_text)[0]
        confidence = max(predicted_proba) * 100
        return predicted_intent[0], confidence
    else:
        predicted_intent = model.predict(transformed_text)
        return predicted_intent[0]


responses = {}
if isinstance(data, dict) and 'data' in data:
    for d in data['data']:
        if isinstance(d, dict) and 'intent' in d and 'answers' in d:
            responses[d['intent']] = d['answers']

last_question = None
last_response = None


def calculate_similarity(text1, text2):
    # Calculate the similarity between two texts using cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    similarity = similarity_matrix[0][0]
    return similarity


def check_keywords(text):
    # Check if the text contains any of the specified keywords
    keywords = ['he', 'she', 'that', 'it', 'who is', 'when it is', 'how long does it take', 'do i need it', 'but',
                'and', 'who is she', 'who is he']
    for keyword in keywords:
        if keyword in text:
            return True
    return False


def generate_response(input_text):
    if isinstance(input_text, list):
        # Combine all questions in the dialogue into a single string
        input_text = ' '.join(input_text)

    # Process the input text and predict the intent
    processed_text = preprocess_text(input_text)
    intent, confidence = predict_intent(processed_text, return_confidence=True)

    # Check if the intent is above a certain confidence threshold
    if confidence > 1.5:
        if intent in responses:
            tag = intent
            response = random.choice(responses[tag])
            # Add context to the response if necessary
            if response == "I'm sorry, I didn't understand your question." and last_question is not None:
                response = "I'm sorry, I don't have an answer to your question about {}. Can I help you with something else?".format(last_question)
            return response

    return "I'm sorry, I didn't understand your question."


def respond(dialogue):
    global last_question, last_response
    # Predict the intent using the model
    if last_question is not None:

        keywords_present = check_keywords(user_input)
        if  keywords_present:
            similarity = calculate_similarity(user_input, last_question)
            if similarity>5:
             dialogue.append(user_input)  # Add the user input to the dialogue

    # Generate response based on the updated dialogue or user input alone
    if len(dialogue) < 4:
        response = generate_response(dialogue)
    else:
        response = generate_response(user_input)

    # Update the context variables
    last_question = user_input
    last_response = response
    return response


# Main conversation loop
dialogue = []  # Variable to store the dialogue questions
print("Bot: Hello! How can I assist you today?")
while True:
    user_input = input("User: ")
    if user_input.lower() == 'goodbye':
        print("Bot: Goodbye!")
        break

    # Check if the user input should be added to the dialogue and get the response
    response = respond(dialogue + [user_input])
    if response is None:
        response = respond([user_input])

    if response:
        dialogue.append(user_input)  # Add the user input to the dialogue

    print("Bot:", response)
    if len(dialogue) >= 4:
        dialogue.clear()  # Clear the dialogue list

    print(dialogue)

