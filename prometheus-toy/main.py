from fastapi import FastAPI
import datetime
import time 
import random   
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter, Histogram


app = FastAPI()
Instrumentator().instrument(app).expose(app)
prediction_counter= Counter("predictions_total", "Total number of predictions")
prediction_histogram= Histogram("prediction_duration_seconds", "Time taken for predictions")

version_gauge = Gauge("model_version_current", "Current version of the model in use")
    

@app.on_event("startup")
def startup_event():
    version_gauge.set(1.0)  # Set the model version to 1.0 on startup
    
    

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(user:dict):
    # Simulate a prediction
    before_prediction = time.time()
    rul = random.uniform(0, 125)
    timestamp = datetime.datetime.utcnow().isoformat()
    model_version = "1.0.0"
    after_prediction = time.time()
    prediction_counter.inc()
    prediction_histogram.observe(after_prediction - before_prediction)
    return {"rul": rul, "timestamp": timestamp, "model_version": version_gauge._value.get()}
    

