# Audit-Ml-Automation
Machine learning-powered automation for auditing journal entries and trial balance data
## Description
This project uses machine learning to detect unusual journal entries and integrates with Power BI for dashboard reporting.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocess and model training scripts.
3. Launch the API: `python api/app.py`
4. Connect Power BI to MySQL to visualize flagged entries.

## Structure
- `data/`: Sample journal and trial balance files
- `models/`: Saved machine learning model
- `scripts/`: Preprocessing, training, and database scripts
- `api/`: Flask app serving predictions
- `dashboard/`: Power BI dashboard
