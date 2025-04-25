from sklearn.ensemble import IsolationForest
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os
from utilis import save_model

def train_isolation_forest(X_train, contamination=0.01, random_state=42):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
    model.fit(X_train)
    return model

def apply_anomaly_detection(model, X_test, original_df, output_path):
    # Predict anomalies
    original_df = original_df.loc[X_test.index].copy()  # only score the test set
    original_df['ANOMALY_SCORE'] = model.decision_function(X_test)
    original_df['IS_ANOMALY'] = model.predict(X_test)
    # Convert to binary (1 = anomaly, 0 = normal)
    original_df['IS_ANOMALY'] = (original_df['IS_ANOMALY'] == -1).astype(int)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_df.to_csv(output_path, index=False)
    print(f"Anomaly detection results saved to: {output_path}")

    # Save trained model with version
    save_model(iso_forest_model)

    return original_df # optionally return for further analysis

if __name__ == "__main__":
    # Load data from previous step
    input_csv = "data/Feature_Engineered_Journal_Data.csv"
    output_csv = "output/Anomaly_Detection_IsolationForest.csv"

    from feature_selector import load_and_select_features
    X, df_all, X_train, X_test = load_and_select_features(input_csv)

    # Train and apply model
    iso_forest_model = train_isolation_forest(X_train)
    anomaly_df = apply_anomaly_detection(iso_forest_model, X_test, df_all, output_csv)

    # Display top anomalies
    anomalies = anomaly_df[anomaly_df["IS_ANOMALY"] == 1]
    print(anomalies.value_counts().head())











































































































































