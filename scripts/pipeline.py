import os
import argparse
import logging
import pandas as pd
import joblib
import shutil

from preprocessing import preprocess_journal_data
from feature_engineering import engineer_features
from feature_selector import load_and_select_features

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info(f"✅ Model loaded from: {model_path}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load model from {model_path}: {e}")
        raise

def apply_model_to_new_data(model, df_features, df_original, output_path):
    df_original = df_original.loc[df_features.index].copy()
    df_original['ANOMALY_SCORE'] = model.decision_function(df_features)
    df_original['IS_ANOMALY'] = model.predict(df_features)
    df_original['IS_ANOMALY'] = (df_original['IS_ANOMALY'] == -1).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_original.to_csv(output_path, index=False)
    logging.info(f"📄 Scored data saved to: {output_path}")

    return df_original

def cleanup_temp_dir(temp_dir):
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"🧹 Temporary directory '{temp_dir}' cleaned up.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to clean temporary files: {e}")

def run_full_pipeline(input_csv, model_path, output_path, keep_temp=False):
    logging.info("🚀 Starting automated anomaly detection pipeline...")

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Step 1: Preprocessing
        temp_cleaned_path = os.path.join(temp_dir, "cleaned_input.csv")
        clean_df = preprocess_journal_data(input_csv, temp_cleaned_path)

        # Step 2: Feature Engineering
        temp_engineered_path = os.path.join(temp_dir, "feature_engineered.csv")
        engineered_df = engineer_features(clean_df, temp_engineered_path)

        # Step 3: Feature Selection
        X, df_cleaned_for_model, _, _ = load_and_select_features(temp_engineered_path)

        # Step 4: Load Model
        model = load_model(model_path)

        # Step 5: Apply Model
        scored_df = apply_model_to_new_data(model, X, df_cleaned_for_model, output_path)

        # Step 6: Show Top Anomalies
        logging.info("\n🚨 Top flagged anomalies:")
        logging.info(f"\n{scored_df[scored_df['IS_ANOMALY'] == 1].head()}")

        logging.info("\n✅ Pipeline complete.")

    finally:
        if not keep_temp:
            cleanup_temp_dir(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Audit Anomaly Detection Pipeline")
    parser.add_argument("--input_csv", required=True, help="Path to input journal entries CSV file")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.joblib file)")
    parser.add_argument("--output_path", required=True, help="Path to save the scored output CSV")
    parser.add_argument("--keep_temp", action="store_true", help="Keep intermediate temp files for debugging")

    args = parser.parse_args()

    run_full_pipeline(
        input_csv=args.input_csv,
        model_path=args.model_path,
        output_path=args.output_path,
        keep_temp=args.keep_temp
    )
