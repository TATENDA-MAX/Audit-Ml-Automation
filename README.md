🧾 Audit ML Automation
A Machine Learning-driven pipeline to automatically detect anomalies in journal entries and generate detailed audit tests for auditors.
Designed for high interactivity with Power BI, minimal manual coding for the audit team, and scalable deployment.

🛠 Project Structure
Folder/File	Purpose
/data	Input journal entries and generated audit test results
/models	Pre-trained anomaly detection models (.joblib)
/output	Scored outputs with anomaly flags
/scripts	Core Python pipeline scripts
/temp	Temporary files (auto-cleaned unless debug mode)
README.md	Project documentation
⚙️ How to Run the Full Pipeline
1. Prerequisites
Python 3.8+ installed

Install project requirements (if any, otherwise skip)

bash
Copy code
pip install -r requirements.txt
(Coming soon: requirements.txt)

2. Execute the Main Pipeline
From project root, run:

bash
Copy code
py scripts/pipeline.py --input_csv data/Unseen_Data.csv --model_path models/isolation_forest_v20250424_164457.joblib --output_path output/Results.csv
3. What Happens Internally
Step	Action
1️⃣	Preprocessing of raw journal data
2️⃣	Feature engineering and enrichment
3️⃣	Feature selection based on trained model
4️⃣	Loading of Isolation Forest model
5️⃣	Scoring journal entries for anomalies
6️⃣	Saving scored results to output/Results.csv
7️⃣	Automatically generating audit test CSVs in data/test_results/
8️⃣	Optional temp folder cleanup
📂 Output Files
File/Folder	Description
output/Results.csv	Full dataset with ANOMALY_SCORE and IS_ANOMALY columns
data/test_results/test_seldom_used_accounts.csv	Journal entries in seldom-used accounts
data/test_results/test_unrelated_account_combinations.csv	Entries involving rare account pairings
data/test_results/test_infrequent_user_entries.csv	Entries posted by infrequent users
data/test_results/test_weekend_entries.csv	Journal entries posted during weekends
data/test_results/test_large_value_entries.csv	Journal entries with unusually large amounts
📊 Power BI Integration
Highly recommended for visualization.

Connect Power BI to:

output/Results.csv (for full anomaly insights)

data/test_results/*.csv (for individual test dashboards)

Build KPI cards, bar charts, line graphs, and filters

Enable audit teams to drill down interactively 🔥

🚀 Future Enhancements
Deploy pipeline as a REST API (FastAPI / Flask)

Add more audit tests (e.g., duplicate entries, post-period adjustments)

Integrate Power BI automatic refresh

Implement email alerts for critical anomalies

Version control models with MLflow

🤝 Contributions
This is a school project 

📜 License
Private corporate project. Internal use only.
