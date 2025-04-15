from flask import Flask, request, jsonify
import joblib, pandas as pd


app = Flask(__name__)
model = joblib.load('../models/anomaly_model.pkl')


@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.DataFrame(request.json)
    df['amount_abs'] = df['amount'].abs()
    df['anomaly_score'] = model.predict(df[['amount_abs']])
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(port=5000)