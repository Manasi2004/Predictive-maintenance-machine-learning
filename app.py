from flask import Flask
from joblib import load

app = Flask(__name__)

@app.route('/')
def load_model():
    global gbm_model
    gbm_model = load(r"C:\Users\856ma\Documents\FlaskApp\trained_model.pkl")
    return 'Model loaded successfully!'

if __name__ == '__main__':
    app.run(debug=True)