from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import joblib

app = FastAPI(
    title="Breast Cancer Classifier",
    description="Classify if a breast cancer sample is malignant or benign.",
    version="1.0"
)

class request_body(BaseModel):
    features: list[float]  # example: [17.99, 10.38, 122.8, 1001.0, ...]

@app.on_event('startup')
def load_model():
    global model
    model = joblib.load("mlruns/1/0a2e9ca4917b4bd48aec7e9bc8b29f98/model/model.pkl") 

@app.get('/')
def root():
    return {'message': 'This is a breast cancer classification model served via MLflow'}

@app.post('/predict')
def predict(data: request_body):
    X = pd.DataFrame([data.features])
    prediction = model.predict(X)
    return {'prediction': int(prediction[0])}
