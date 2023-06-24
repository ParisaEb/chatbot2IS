import json
import re
import nltk
import sqlite3
from datetime import date
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import email.utils
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
nltk.download('punkt')
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
with open('intentsComplete.json', encoding='utf-8') as file:
    data = json.load(file)


# Connect to the SQLite database
current_date = str(date.today())
#conn = sqlite3.connect('chatbot_Database.db')
conn = sqlite3.connect('chatbot_Database.db', check_same_thread=False)
conn.row_factory = sqlite3.Row

# Get a connection from the pool
def get_connection():
    return conn



# Create a table
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users
                  (email TEXT, Email_confirmed BOOLEAN, date_of_use TEXT)''')

cursor.execute(''' CREATE TABLE IF NOT EXISTS TEACHERS (email TEXT, position TEXT,name TEXT, courses TEXT)''')
conn.commit()


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
    stop_words.remove("who")
    stop_words.remove("how")
    stop_words.remove("or")
    stop_words.add("tell")
    word_tokens = nltk.word_tokenize(text)
    # Remove punctuation except for "."
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


def check_Responces(last_response,moreDetailedResponse):
  if (last_response==moreDetailedResponse):
    return False
  return True


last_question = None
last_response = None

user_email =None
Email_confirmed = 0
user_date_of_use = str(date.today())



def respond(user_input):
    global last_question, last_response, last_processed_text, current_processed_text
    current_processed_text = preprocess_text(user_input)
    conn = get_connection()
    cursor = conn.cursor()
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_regex, user_input)

    # Check if the user input is a clarification
    clarification_phrases=["well I try to explain more ", "let me put it this way  ","sorry for confusion let me explain more "]

    if predict_intent(user_input) == "user.clarification" and last_response is not None:
        if last_question is not None:
            tag = predict_intent(last_question)
            if tag in responses:
                available_answers = [answer for answer in responses[tag] if answer != last_response]
                if available_answers:
                    response = random.choice(clarification_phrases) + random.choice(available_answers)
                    last_response = response
                    return response

    if email_matches:
        user_email = email_matches[0]

        # Validate the email address
        is_valid = email.utils.parseaddr(user_email)[1] != ''

        if not is_valid:
            return "Please provide a valid email address."

        # Check if the email exists in the teachers table
        cursor.execute("SELECT * FROM teachers WHERE email=?", (user_email,))
        teacher_data = cursor.fetchone()
        if teacher_data:
            return f"Teacher information: Position: {teacher_data[1]}, Courses: {teacher_data[3]}"
        else:
            # Email not found in the teachers table, insert into users table
            cursor.execute("INSERT INTO users (email, Email_confirmed, date_of_use) VALUES (?, ?, ?)",
                           (user_email, Email_confirmed, user_date_of_use))
            conn.commit()
            return "Thank you for providing us with your email address!"
    else:
        # Process the input using predict_intent and the rest of the code
        intent, confidence = predict_intent(user_input, return_confidence=True)
        print(confidence)

        # Check if the intent is above a certain confidence threshold
        if confidence > 1.30:
            if intent in responses:
                tag = intent
                response = random.choice(responses[tag])
                # Update the context variables
                last_question = user_input
                last_response = response
                return response

    if len(current_processed_text) < 2 and user_input:
        return "I'm sorry, but I need more context to accurately answer your question. " \
               "Could you provide more information or clarify what you are referring to?"


    return "I'm sorry, but I didn't understand your question."
    # Predict the intent using the model
    print(confidence)

# Main conversation loop
dialogue = []  # Variable to store the dialogue questions
last_question = ''

print("Bot: Hello! How can I assist you today?")
while True:
    user_input = input("User: ")
    dialogue.append(user_input)

    if user_input.lower() == 'goodbye':
        print("Bot: Goodbye!")
        break
    else:

        response = respond(user_input)

    print("Bot:", response)

    last_question = user_input


    #if len(dialogue) >= 4:
      #  dialogue.clear()  # Clear the dialogue list
    #print(dialogue)
    #print(last_question)
    print(predict_intent(user_input))
    print("the processed text is", preprocess_text(user_input))

