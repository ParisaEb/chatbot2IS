from flask import Flask, render_template

app = Flask(__name__, static_folder='Static', template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
def home():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()