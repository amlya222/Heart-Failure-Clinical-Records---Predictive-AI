from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for local HTML

# Load the dataset
DATA_PATH = 'heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(DATA_PATH)

# Prepare data for modeling
FEATURES = [col for col in df.columns if col not in ['DEATH_EVENT']]
X = df[FEATURES]
y = df['DEATH_EVENT']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

@app.route('/api/summary', methods=['GET'])
def summary():
    # Return basic statistics
    summary_stats = df.describe().to_dict()
    return jsonify(summary_stats)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    # Ensure all features are present
    try:
        input_features = np.array([[data[feature] for feature in FEATURES]])
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    input_scaled = scaler.transform(input_features)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0, 1]
    return jsonify({'prediction': int(pred), 'probability': float(prob)})

if __name__ == '__main__':
    app.run(debug=True) 