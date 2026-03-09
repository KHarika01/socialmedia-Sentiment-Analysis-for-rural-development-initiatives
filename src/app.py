from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Find project base folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
model = pickle.load(open(os.path.join(BASE_DIR, "Models", "naive_bayes.pkl"), "rb"))

# Load vectorizer
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "Models", "vectorizer.pkl"), "rb"))

@app.route("/")
def home():
    return "Flask Backend Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    text = data["text"]

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()

    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(confidence),2)
    })

if __name__ == "__main__":
    app.run(port=5000)