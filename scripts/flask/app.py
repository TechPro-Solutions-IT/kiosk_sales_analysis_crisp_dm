from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load the model
# model = joblib.load('../model/best_model.pkl')

# Define the relative path to the model file
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model', 'best_model.pkl')

# Load the model
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Welcome to the Sales Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])
        # Predict using the loaded model
        prediction = model.predict(input_data)
        # Return the prediction result
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
