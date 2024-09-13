# Chatbot with SQLite Database and Intent Classification
This project is a chatbot application that utilizes natural language processing (NLP) techniques to classify user intents and provide relevant responses. The chatbot is built using Python, NLTK, SQLite, and Flask, allowing it to process user inputs, store user data in a database, and provide dynamic, context-aware responses.

# Project Overview
The chatbot performs the following tasks:

# Intent Classification: Uses NLP techniques to classify user inputs into predefined intents using a Linear Support Vector Classifier (SVC).
Database Interaction: Stores and retrieves user data from an SQLite database.
Dynamic Responses: Provides dynamic responses based on the identified intent, with a fallback for low-confidence predictions.
Web Interface: A Flask-based web application that allows users to interact with the chatbot via a simple web interface.
Prerequisites
To run the project, ensure you have the following Python packages installed:

Python 3.6 or higher
NLTK
SQLite3
Flask
Scikit-learn
You can install the required packages using the following command:

bash
Copy code
pip install nltk sqlite3 flask scikit-learn
Dataset
The chatbot uses a JSON file (intentsComplete.json) containing predefined intents and their corresponding utterances and responses. The data is loaded and processed to create the necessary features for intent classification.

How It Works
NLP and Intent Classification
The chatbot uses NLTK to tokenize and stem user inputs. It then uses a CountVectorizer and TfidfTransformer to transform the input text into numerical features, which are then classified using a Linear Support Vector Classifier (SVC).

Example of preprocessing and classification:

python
Copy code
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

def predict_intent(text, return_confidence=False):
    processed_text = preprocess_text(text)
    transformed_text = tfidf_transformer.transform(vectorizer.transform([processed_text]))
    predicted_intent = model.predict(transformed_text)
    return predicted_intent[0]
Database Interaction
The chatbot connects to an SQLite database where it stores user information, such as email addresses and user inputs. It also retrieves teacher information if the user’s email matches a record in the database.

Example of database interaction:

python
Copy code
# Connect to the SQLite database
conn = sqlite3.connect('chatbot_Database.db', check_same_thread=False)

# Create a table
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT, Email_confirmed BOOLEAN, date_of_use TEXT)''')
conn.commit()
Flask Web Interface
The chatbot is integrated into a Flask web application, allowing users to interact with it through a web interface.

Example of Flask setup:

python
Copy code
from flask import Flask, render_template, jsonify, request
from main import respond

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = respond(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
Usage
Run the Chatbot: Start the chatbot by running the Flask app. Navigate to the home page to start interacting with the bot.
Input Processing: The chatbot will process user inputs, predict the intent, and return a relevant response.
Database Operations: The chatbot will store user data in the SQLite database and retrieve teacher information when applicable.
Future Enhancements
Advanced NLP: Implement advanced NLP techniques for better intent classification and entity recognition.
User Management: Add features for user authentication and personalized responses.
Deployment: Deploy the chatbot to a cloud platform for wider accessibility.
License
This project is licensed under the MIT License.
