import json
import joblib
import numpy as np

def init():
    global model, vectorizer
    model = joblib.load("bbc_logistic_model.pkl")
    vectorizer = joblib.load("bbc_tfidf_vectorizer.pkl")

def run(raw_data):
    try:
        data = json.loads(raw_data)["text"]
        if isinstance(data, str):
            data = [data]
        vectorized = vectorizer.transform(data)
        prediction = model.predict(vectorized)
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
