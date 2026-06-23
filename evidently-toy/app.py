from fastapi import FastAPI
from prometheus_client import Gauge
from detector import detect_drift
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

drift_score_current= Gauge('psi_score', 'Indicates the current PSI score for the dataset')


df = pd.read_csv("./data/train_FD001.txt", sep=r'\s+', header=None, names = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3", "sensor_measurement_1", "sensor_measurement_2", "sensor_measurement_3", "sensor_measurement_4", "sensor_measurement_5", "sensor_measurement_6", "sensor_measurement_7", "sensor_measurement_8", "sensor_measurement_9", "sensor_measurement_10", "sensor_measurement_11", "sensor_measurement_12", "sensor_measurement_13", "sensor_measurement_14", "sensor_measurement_15", "sensor_measurement_16", "sensor_measurement_17", "sensor_measurement_18", "sensor_measurement_19", "sensor_measurement_20", "sensor_measurement_21"])

df = df.drop(columns=["unit_number", "time_in_cycles"])

split_id = int(len(df) * 0.8)
reference_df = df.iloc[:split_id]
current_df = df.iloc[split_id:]

@app.get("/detect_drift")
def detect_drift_endpoint():
    result = detect_drift(reference_df, current_df)
    drift_score_current.set(result["aggregate_drift_score"])
    return result


@app.get("/health")
def health_check():
    return {"status": "healthy"}




































































