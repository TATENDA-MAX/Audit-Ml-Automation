import pandas as pd
import mysql.connector
from joblib import load

def save_anomalies_to_mysql(data_path, model_path):
    print("📦 Loading cleaned data...")
    df = pd.read_csv(data_path)

    print("🧠 Loading trained model...")
    model = load(model_path)

    print("🔍 Predicting anomalies...")
    df['Anomaly'] = model.predict(df[['NetAmount']])
    df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})  # 1 = anomaly


    anomalies = df[df['Anomaly'] == 1]
    print(f"🚨 Total anomalies detected: {len(anomalies)}")

    print("🗃️ Saving anomalies to MySQL...")

    connection = mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "Luftwaffer1",
        database = "tatenda_test",
    )
    cursor = connection.cursor()

    # Create table if it doesn't exist

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS anomalies (
                   id INT AUTO_INCREMENT PRIMARY KEY,
                   Description TEXT, 
                   Debit FLOAT,
                   Credit FLOAT,
                   NetAmount FLOAT
                   )
            """)
    
    for _, row in anomalies.iterrows():
        cursor.execute("""
                       INSERT INTO anomalies (Description, Debit, Credit, NetAmount)
                       VALUES (%s, %s, %s, %s)
                   """, (row['Description'], row['Debit'], row['Credit'], row['NetAmount']))
        
    connection.commit()
    cursor.close()
    connection.close()

    print("✅ Anomalies saved successfully!")


if __name__ == "__main__":
    data_file = "data/cleaned_journal_entries.csv"
    model_file = "models/isolation_forest_model.pkl"
    save_anomalies_to_mysql(data_file, model_file)