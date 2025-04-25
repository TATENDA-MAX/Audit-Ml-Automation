import pandas as pd
import joblib
import os
from feature_engineering import engineer_features
from preprocessing import preprocess_journal_data
from feature_selector import load_and_select_features

def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model Loaded from: {model_path}")
    return model

def load_new_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"New data loaded from: {csv_path}")
    return df

def apply_model_to_new_data(model, df_features, df_original, output_path):
    # Score data
    df_original = df_original.loc[df_features.index].copy()
    df_original['ANOMALY_SCORE'] = model.decision_function(df_features)
    df_original['IS_ANOMALY'] = model.predict(df_features)
    df_original['IS_ANOMALY'] = (df_original['IS_ANOMALY'] == -1).astype(int)


    # Save results

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_original.to_csv(output_path, index=False)
    print(f"Scored new data saved to: {output_path}")
    return df_original

if __name__ == "__main__":
    # === CONFIG ===
    model_path = "models/isolation_forest_v20250424_164457.joblib"
    new_data_path = "data/Unseen_Data.csv"
    output_path = "output/Scored_New_Journal_Entries.csv"


    # Load model and new data
    model = load_model(model_path)
    df_new = load_new_data(new_data_path)

    # Preprocess data
    temp_output_path = "temp/cleaned_temp.csv"
    os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
    Clean_data = preprocess_journal_data(new_data_path, temp_output_path)

    # Feature engineering the data
    feature_engineered_path = "temp/feature_engineered_temp.csv"
    feature_Engineered_data = engineer_features(Clean_data, feature_engineered_path)

    # Feature selection for scoring

    X, df_with_all_fields, X_train, X_test = load_and_select_features(feature_engineered_path)


    # Score new data
    scored_df = apply_model_to_new_data(model, X, df_with_all_fields, output_path)

    # Show sample of flagged anomalies
    print("\nTop anomalies:")
    print(scored_df[scored_df["IS_ANOMALY"] == 1].head())
