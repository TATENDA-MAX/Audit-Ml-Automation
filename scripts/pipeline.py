# pipeline.py

import os
import pandas as pd
import joblib
from preprocessing import preprocess_journal_data
from feature_engineering import engineer_features
from feature_selector import load_and_select_features

def load_model(model_path):
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model

def apply_model_to_new_data(model, df_features, df_original, output_path):
    df_original = df_original.loc[df_features.index].copy()
    df_original['ANOMALY_SCORE'] = model.decision_function(df_features)
    df_original['IS_ANOMALY'] = model.predict(df_features)
    df_original['IS_ANOMALY'] = (df_original['IS_ANOMALY'] == -1).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_original.to_csv(output_path, index=False)
    print(f"📄 Scored data saved to: {output_path}")

    return df_original

def run_full_pipeline(input_csv, model_path, output_path):
    print("🚀 Starting automated anomaly detection pipeline...")

    # Step 1: Preprocessing
    temp_cleaned_path = "temp/cleaned_input.csv"
    os.makedirs(os.path.dirname(temp_cleaned_path), exist_ok=True)
    clean_df = preprocess_journal_data(input_csv, temp_cleaned_path)

    # Step 2: Feature Engineering
    temp_engineered_path = "temp/feature_engineered.csv"
    engineered_df = engineer_features(clean_df, temp_engineered_path)

    # Step 3: Feature Selection
    X, df_cleaned_for_model, _, _ = load_and_select_features(temp_engineered_path)

    # Step 4: Load Model
    model = load_model(model_path)

    # Step 5: Apply Model
    scored_df = apply_model_to_new_data(model, X, df_cleaned_for_model, output_path)

    # Step 6: Show Top Anomalies
    print("\n🚨 Top flagged anomalies:")
    print(scored_df[scored_df["IS_ANOMALY"] == 1].head())

    print("\n✅ Pipeline complete.")

if __name__ == "__main__":
    input_csv = "data/Unseen_Data.csv"
    model_path = "models/isolation_forest_v20250424_164457.joblib"
    output_path = "output/Scored_New_Journal_Entries.csv"
    
    run_full_pipeline(input_csv, model_path, output_path)
