from sklearn.ensemble import IsolationForest
import pandas as pd
from joblib import dump

def train_model(data_path, model_path):
    print("✅ Loading data...")
    df = pd.read_csv(data_path)

    # Train on NetAmount column only
    X = df[['NetAmount']]

    print("✅ Training Isolation Forest model...")
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X)

    # Save the model
    dump(model, model_path)
    print(f"✅ Model trained and saved to: {model_path}")

if __name__ == "__main__":
    input_csv = "data/cleaned_journal_entries.csv"
    model_file = "models/isolation_forest_model.pkl"
    train_model(input_csv, model_file)
