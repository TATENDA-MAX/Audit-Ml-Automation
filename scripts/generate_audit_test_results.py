import pandas as pd
import os

def save_filtered_csv(df, condition, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[condition].to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def generate_test_results(input_path, output_dir):
    df = pd.read_csv(input_path, dtype={0: str}, low_memory = False)


    # Test 1: Seldom used Acoounts
    save_filtered_csv(df, df['ACCOUNT_USAGE_COUNT'] < 5, f"{output_dir}/test_seldom_used_accounts.csv")

    # Test 2: Unrelated Account Combinations (rare pairs)
    save_filtered_csv(df, df['TRANSACTION_ACCOUNT_PAIR_COUNT'] <= 1, f"{output_dir}/test_unrelated_account_combinations.csv")

    # Test 3: Entries by Infrequent Users 
    save_filtered_csv(df, df['USER_ACTIVITY_LEVEL'] < 5, f"{output_dir}/test_infrequent_user_entries.csv")

    # Test 4: Weekend Entries
    save_filtered_csv(df, df['WEEKEND_POSTING'] == True, f"{output_dir}/test_weekend_entries.csv")

    # Test 5: Large Value Entries
    save_filtered_csv(df, df['IS_LARGE_VALUE'] == 1, f"{output_dir}/test_large_value_entries.csv")

if __name__ == "__main__":
    input_csv = "output/Anomaly_Detection_IsolationForest.csv"
    output_dir = "data/test_results"
    generate_test_results(input_csv, output_dir)