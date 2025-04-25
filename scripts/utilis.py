import os
import joblib
from datetime import datetime

def save_model(model, model_dir="models", version_tag=None):
    os.makedirs(model_dir, exist_ok=True)
    if version_tag is None:
        version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"isolation_forest_v{version_tag}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return model_path