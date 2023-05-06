# Import libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import *
import keras
import joblib
import seaborn as sns
from flask import Flask, request, jsonify
import pickle
import os

# Create a Flask app instance
app = Flask(__name__)

# Get model path and load model
path = os.getcwd()
file_name = path + 'hard_drive_model.pkl'
xgb_model_loaded = pickle.load(open(file_name, "rb"))

# Column names
column_names = ["capacity_bytes",	"smart_1_normalized",	"smart_1_raw",	"smart_3_normalized",	"smart_3_raw",	"smart_4_raw",	"smart_5_raw",	
                "smart_7_normalized",	"smart_9_normalized",	"smart_9_raw",	"smart_12_raw",	"smart_194_normalized",	"smart_194_raw",
                "smart_197_raw", "smart_199_raw", "useful_life"]

## Flask App ##

# Define the API endpoint and request method
@app.route('/predict', methods=['POST'])
def predict():
    # Get the incoming data from the request
    data = request.get_json()

    # Convert the data into a DataFrame
    sample = pd.DataFrame(data, columns=column_names)

    # Transform the data and get the prediction
    X = sample.drop(columns = ['useful_life'], axis = 1)
    prediction = xgb_model_loaded.predict(X, verbose=False)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)