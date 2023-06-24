<<<<<<< HEAD
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
=======
from flask import Flask, render_template,jsonify,request


from main import generate_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return {'response': response}

if __name__ == '__main__':
>>>>>>> 2cadfeb0af745c3c7e6d817576e5f3787bdfe902
    app.run()