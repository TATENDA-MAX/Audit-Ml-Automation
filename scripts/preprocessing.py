import pandas as pd
import numpy as np
import os

def preprocess_journal_data(input_path, output_path=None):
    # Load Data
    df = pd.read_csv(input_path, dtype={0: str}, low_memory=False)

    # Standardize column names
    df.columns = [col.strip().upper().replace(' ', '_') for col in df.columns]

    # Convert dates to datetime format
    df['POSTING_DATE'] = pd.to_datetime(df['POSTING_DATE'], errors='coerce', dayfirst=True)
    df['EFFECTIVE_DATE'] = pd.to_datetime(df['EFFECTIVE_DATE'], errors='coerce', dayfirst=True)

    # Convert SYSTEM_VALUE to numeric
    df['SYSTEM_VALUE'] = pd.to_numeric(df['SYSTEM_VALUE'], errors='coerce')

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows with missing key fields
    df = df.dropna(subset=['ACCOUNT_NUMBER', 'SYSTEM_VALUE', 'POSTING_DATE'])

    # Feature Engineering
    df['YEAR'] = df['POSTING_DATE'].dt.year
    df['MONTH'] = df['POSTING_DATE'].dt.month
    df['DAY'] = df['POSTING_DATE'].dt.day

    # Strip text fields
    df['USER_NAME'] = df['USER_NAME'].astype(str).str.strip().str.upper()
    df['ENTRY_SOURCE'] = df['ENTRY_SOURCE'].astype(str).str.strip().str.upper()
    df['ACCOUNT_ENTRY_DESCRIPTION'] = df['ACCOUNT_ENTRY_DESCRIPTION'].astype(str).str.strip()

    df_cleaned = df.copy()

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

    return df_cleaned
if __name__ == "__main__":
    input_csv = "data/Sample Journal Dump.csv"
    output_csv = "data/Clean_journal_data.csv"
    preprocess_journal_data(input_csv, output_csv)
