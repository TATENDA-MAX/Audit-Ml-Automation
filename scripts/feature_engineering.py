import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ----------------------
# Time-Based Features
# ----------------------
def add_time_features(df):
    df['POSTING_DAY_OF_WEEK'] = df['POSTING_DATE'].dt.day_name()
    df['WEEKEND_POSTING'] = df['POSTING_DATE'].dt.weekday >= 5
    df['DAYS_BETWEEN_POST_AND_EFFECTIVE'] = (df['POSTING_DATE'] - df['EFFECTIVE_DATE']).dt.days
    return df

# ----------------------
# Monetary Features
# ----------------------
def add_monetary_features(df):
    df['ABS_VALUE'] = df['SYSTEM_VALUE'].abs()
    df['IS_LARGE_VALUE'] = (df['ABS_VALUE'] > df['ABS_VALUE'].quantile(0.99)).astype(int)
    return df

# ----------------------
# Frequency-Based Features
# ----------------------
def add_frequency_features(df):
    df['ACCOUNT_USAGE_COUNT'] = df['ACCOUNT_NUMBER'].map(df['ACCOUNT_NUMBER'].value_counts())
    df['USER_ACTIVITY_LEVEL'] = df['USER_NAME'].map(df['USER_NAME'].value_counts())
    return df

# ----------------------
# Text-Based Features
# ----------------------
def add_text_features(df):
    df['DESC_LENGTH'] = df['ACCOUNT_ENTRY_DESCRIPTION'].astype(str).str.len()
    df['HAS_KEYWORDS'] = df['ACCOUNT_ENTRY_DESCRIPTION'].str.contains(r"adjustment|manual|reversal|Abuse|Bribe|as directed|as requested|Query|conceal|confident|corrupt|coverup|cushion|deterioration|Opportunity|Dummy|Early|Fraud|Hidden|Immaterial|remove|restate|test|related party|kickback|override|manipulate", case=False, na=False).astype(int)
    return df

# ----------------------
# User Behavior Features
# ----------------------
def add_user_behavior_features(df):
    daily_user_activity = df.groupby(['USER_NAME', 'POSTING_DATE']).size()
    df['DAILY_ENTRIES_BY_USER'] = df.set_index(['USER_NAME', 'POSTING_DATE']).index.map(daily_user_activity)
    user_value_std = df.groupby('USER_NAME')['SYSTEM_VALUE'].transform('std')
    df['USER_VALUE_STD'] = user_value_std
    return df

# ----------------------
# Account Relationship Features
# ----------------------
def add_account_relationship_features(df):
    pair_freq = df.groupby(['HEADER_NUMBER', 'ACCOUNT_NUMBER']).size()
    df['TRANSACTION_ACCOUNT_PAIR_COUNT'] = df.set_index(['HEADER_NUMBER', 'ACCOUNT_NUMBER']).index.map(pair_freq)
    return df

# ----------------------
# Categorical Encoding
# ----------------------
def add_categorical_encodings(df):
    le_source = LabelEncoder()
    le_user = LabelEncoder()
    df['ENTRY_SOURCE_ENC'] = le_source.fit_transform(df['ENTRY_SOURCE'].astype(str))
    df['USER_NAME_ENC'] = le_user.fit_transform(df['USER_NAME'].astype(str))
    return df

# ----------------------
# Master Function
# ----------------------
def engineer_features(input_data, output_path=None):
    # Handle if input is a path or a Dataframe
    if input_data is None:
        raise ValueError("Input data to 'engineer_features()' cannot be None.")
    if isinstance(input_data, str):

        df = pd.read_csv(input_data, parse_dates=['POSTING_DATE', 'EFFECTIVE_DATE'], dtype={0:str}, low_memory=False)
    else:
        df = input_data.copy()
        df['POSTING_DATE'] = pd.to_datetime(df['POSTING_DATE'], errors='coerce')
        df['EFFECTIVE_DATE'] = pd.to_datetime(df['EFFECTIVE_DATE'], errors='coerce')

    df = add_time_features(df)
    df = add_monetary_features(df)
    df = add_frequency_features(df)
    df = add_text_features(df)
    df = add_user_behavior_features(df)
    df = add_account_relationship_features(df)
    df = add_categorical_encodings(df)

    # Save if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Feature-engineered data saved to: {output_path}")

    return df

if __name__ == "__main__":
    input_csv = "data/Clean_journal_data.csv"
    output_csv = "data/Feature_Engineered_Journal_Data.csv"
    engineer_features(input_csv, output_csv)
