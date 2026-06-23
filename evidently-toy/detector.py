import pandas as pd
from evidently import Report

from evidently.presets import DataDriftPreset


df = pd.read_csv("./data/train_FD001.txt", sep=r'\s+', header=None, names = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3", "sensor_measurement_1", "sensor_measurement_2", "sensor_measurement_3", "sensor_measurement_4", "sensor_measurement_5", "sensor_measurement_6", "sensor_measurement_7", "sensor_measurement_8", "sensor_measurement_9", "sensor_measurement_10", "sensor_measurement_11", "sensor_measurement_12", "sensor_measurement_13", "sensor_measurement_14", "sensor_measurement_15", "sensor_measurement_16", "sensor_measurement_17", "sensor_measurement_18", "sensor_measurement_19", "sensor_measurement_20", "sensor_measurement_21"])

df = df.drop(columns=["unit_number", "time_in_cycles"])

split_id = int(len(df) * 0.8)
reference_df = df.iloc[:split_id]
current_df = df.iloc[split_id:]

print("Reference DataFrame shape:", reference_df.shape)
print("Current DataFrame shape:", current_df.shape)

report = Report([DataDriftPreset(method="psi")])

myeval = report.run(reference_df,current_df)

psi_values =[]
per_features = {}

result = myeval.dict()
for metric in result['metrics']:
    if metric['metric_name'].startswith('ValueDrift'):
        
        drift_column = metric['config']['column']
        psi_score = float(metric['value'])

        per_features[drift_column] = psi_score
        psi_values.append(psi_score)


# Aggregate drift score = mean PSI across all features
aggregate_drift_score = sum(psi_values) / len(psi_values)

# Drift detected if any feature PSI > 0.2
drift_detected = any(score > 0.2 for score in psi_values)

print("Drift Detected:", drift_detected)
print("Aggregate Drift Score:", aggregate_drift_score)
print("Per Feature Scores:")
print(per_features)