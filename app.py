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