from flask import Flask, render_template, request
import re
import joblib
import sklearn

app = Flask(__name__)

best_model = joblib.load('svm/svm_model.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def predict():
    review = request.form["review"]
    processed_review = preprocess_text(review)

# 

    prediction = best_model.predict([processed_review])[0]
    sentiment = "Positive" if prediction == "Positive" else "Negative"
    return render_template("results.html", review=review, sentiment=sentiment)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)