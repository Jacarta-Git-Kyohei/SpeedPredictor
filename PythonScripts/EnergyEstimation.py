import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

# ------ Customize Parameters ------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATASET_FILE = os.path.join(SCRIPT_DIR, "..", "CreatedDataset", "SpeedPredictorDataset_Chunk3500.csv")
ENERGY_ESTIMATE_FILE = os.path.join(SCRIPT_DIR, "..", "CreatedDataset", "DatasetForEnergyEstimation.csv")
OUTPUT_ESTIMATE_FILE = os.path.join(SCRIPT_DIR, "..", "CreatedDataset", "SpeedPrediction_WithCalories.csv")
MODEL_FILE = os.path.join(SCRIPT_DIR, "..", "CreatedDataset", "xgb_speed_predictor.pkl")
CHUNK_INTERVAL_MS = 3500
WEIGHT_KG = 70.0
# ----------------------------------------------------------

SPEED_REPR = np.array([3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0, 8.4, 9.7, 10.8])
INTENSITY_REPR = np.array([2.8, 3.0, 3.5, 4.3, 5.0, 7.0, 8.3, 9.0, 9.8, 10.5])

def map_speed_to_intensity(speed_kmh):
    speeds = np.array(speed_kmh, dtype=float)
    idx = np.argmin(np.abs(speeds.reshape(-1,1) - SPEED_REPR.reshape(1,-1)), axis=1)
    return INTENSITY_REPR[idx]

def train_or_load_model(training_csv_path, model_path):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} ...")
        model = joblib.load(model_path)
        return model

    if not os.path.exists(training_csv_path):
        raise FileNotFoundError(f"Neither model file nor training csv found. Missing: {training_csv_path}")

    print(f"Training new XGBoost model using {training_csv_path} ...")
    df = pd.read_csv(training_csv_path)
    candidate_features = ['accel_STD', 'accel_RMS', 'accel_DomFreq1', 'accel_DomFreq2', 'ang_STD', 'ang_RMS', 'ang_DomFreq1', 'ang_DomFreq2']
    features = [c for c in candidate_features if c in df.columns]
    if 'TargetSpeed' not in df.columns:
        raise ValueError("Training csv must contain 'TargetSpeed' column.")
    X = df[features]
    y = df['TargetSpeed']

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    return model

def predict_and_compute_calories(input_csv, model, output_csv, weight_kg, chunk_interval_ms):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input estimation CSV not found: {input_csv}")
    df_est = pd.read_csv(input_csv)
    try:
        feature_names = model.get_booster().feature_names
        if feature_names is None:
            candidate_features = ['accel_STD', 'accel_RMS', 'accel_DomFreq1', 'accel_DomFreq2', 'ang_STD', 'ang_RMS', 'ang_DomFreq1', 'ang_DomFreq2']
            feature_names = [c for c in candidate_features if c in df_est.columns]
    except Exception:
        candidate_features = ['accel_STD', 'accel_RMS', 'accel_DomFreq1', 'accel_DomFreq2', 'ang_STD', 'ang_RMS', 'ang_DomFreq1', 'ang_DomFreq2']
        feature_names = [c for c in candidate_features if c in df_est.columns]

    if len(feature_names) == 0:
        raise ValueError("No feature columns found in estimation CSV that match model features.")

    X_est = df_est[feature_names]

    preds = model.predict(X_est)
    df_est['PredictedSpeed_km_h'] = preds

    intensities = map_speed_to_intensity(preds)
    df_est['Intensity_MET'] = intensities

    seconds_per_row = chunk_interval_ms / 1000.0
    hours_per_row = seconds_per_row / 3600.0

    df_est['Calories_kcal'] = df_est['Intensity_MET'] * hours_per_row * weight_kg

    total_kcal = df_est['Calories_kcal'].sum()
    print(f"Predicted {len(df_est)} rows. Total estimated kcal = {total_kcal:.2f} kcal")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_est.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return df_est

def main():
    model = train_or_load_model(TRAINING_DATASET_FILE, MODEL_FILE)

    df_result = predict_and_compute_calories(ENERGY_ESTIMATE_FILE, model, OUTPUT_ESTIMATE_FILE, weight_kg=WEIGHT_KG, chunk_interval_ms=CHUNK_INTERVAL_MS)

    print("Predicted speed stats (km/h):")
    print(df_result['PredictedSpeed_km_h'].describe())

if __name__ == "__main__":
    main()
