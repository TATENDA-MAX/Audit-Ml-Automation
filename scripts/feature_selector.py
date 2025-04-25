import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_select_features(input_path):
    # Load feature-engineered data
    df = pd.read_csv(input_path, dtype={0:str}, low_memory=False)

    # Define the features to use (you can add/remove as needed)
    selected_features = [
        'YEAR', 'MONTH', 'DAY',
        'WEEKEND_POSTING',
        'DAYS_BETWEEN_POST_AND_EFFECTIVE',
        'ABS_VALUE', 'IS_LARGE_VALUE',
        'ACCOUNT_USAGE_COUNT', 'USER_ACTIVITY_LEVEL',
        'ENTRY_SOURCE_ENC', 'USER_NAME_ENC',
        'TRANSACTION_ACCOUNT_PAIR_COUNT'
    ]

    # Filter out rows with missing values in selected features
    df_cleaned = df.dropna(subset=selected_features).copy()

    # Prepare feature matrix X
    X = df_cleaned[selected_features]

    # Train-test split
    X_train, X_test, df_train, df_test = train_test_split(X, df_cleaned, test_size = 0.3, random_state=42)

    print(f"Loaded {X.shape[0]} records with {X.shape[1]} features for modeling.")
    return X, df_cleaned, X_train, X_test

if __name__ == "__main__":
    input_csv = "data/Feature_Engineered_Journal_Data.csv"
    X, df_with_all_fields, X_train, X_test = load_and_select_features(input_csv)
