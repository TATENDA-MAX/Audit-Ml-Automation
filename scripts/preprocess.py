import pandas as pd

def preprocess_journal_entries(input_path, output_path):

    print(">>> preprocess_journal_entries() called")
    # Load the data

    df = pd.read_csv(input_csv)

    # Convert date fields to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['PostingDate'] = pd.to_datetime(df['PostingDate'])

    # Drop completely empty rows
    df.dropna(how='all', inplace=True)

    # Add NetAmount column
    df['NetAmount'] = df['Debit'] - df['Credit']

    # Save the cleaned version
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    input_csv = "data/sample_journal_entries.csv"
    output_csv = "data/cleaned_journal_entries.csv"
    preprocess_journal_entries(input_csv, output_csv)


          



