# scripts/analyze_results.py

import pandas as pd
import argparse
import os

def analyze_results(results_csv):
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"File not found: {results_csv}")

    df = pd.read_csv(results_csv)
    if 'IS_ANOMALY' not in df.columns or 'ANOMALY_SCORE' not in df.columns:
        raise ValueError("Required columns ('IS_ANOMALY', 'ANOMALY_SCORE') not found in the results file.")

    total_entries = len(df)
    total_anomalies = df['IS_ANOMALY'].sum()
    anomaly_rate = (total_anomalies / total_entries) * 100

    print("\n📊 Basic Statistics:")
    print(f"- Total Journal Entries: {total_entries}")
    print(f"- Total Anomalies: {total_anomalies}")
    print(f"- Anomaly Rate: {anomaly_rate:.2f}%")

    print("\n🚨 Top 10 Anomalous Entries:")
    top_anomalies = df[df['IS_ANOMALY'] == 1].sort_values(by='ANOMALY_SCORE', ascending=False).head(10)
    if not top_anomalies.empty:
        print(top_anomalies[['ANOMALY_SCORE'] + [col for col in df.columns if col not in ('ANOMALY_SCORE', 'IS_ANOMALY')]].to_string(index=False))
    else:
        print("No anomalies detected.")

    print("\n📈 Anomaly Score Summary:")
    print(df['ANOMALY_SCORE'].describe())

def main():
    parser = argparse.ArgumentParser(description="Analyze scored journal entries results.")
    parser.add_argument('--results_csv', required=True, help='Path to the scored results CSV file.')
    args = parser.parse_args()

    analyze_results(args.results_csv)

if __name__ == "__main__":
    main()
