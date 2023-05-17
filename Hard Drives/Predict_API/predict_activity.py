# Import libraries
import pickle
import os
import boto3
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import *
from flask import Flask, request, jsonify

# Create a Flask app instance
app = Flask(__name__)

# Get model path and load model
path = os.getcwd()
file_name = 'xgb_model.json'
xgb_model_loaded = XGBRegressor()
xgb_model_loaded.load_model(file_name)

column_names = ["capacity_bytes",	"smart_1_normalized",\
    	"smart_1_raw",	"smart_3_normalized",	"smart_3_raw",\
        "smart_4_raw",	"smart_5_raw",	"smart_7_normalized",	"smart_9_normalized",\
    	"smart_9_raw",	"smart_12_raw",	"smart_194_normalized",	"smart_194_raw",\
        "smart_197_raw", "smart_199_raw"]

## Flask App ##

# Define the API endpoint and request method
@app.route('/predict', methods=['POST'])
def predict():
    # Get the incoming data from the request
    data = request.get_json()
    data1 = {col: data[0][col] for col in column_names}
    data2 = {i: int(data1[i]) for i in list(data1.keys())}

    # Convert the data into a DataFrame
    sample = pd.DataFrame([data2], columns=column_names)

    # Transform the data and get the prediction
    prediction = xgb_model_loaded.predict(sample)

    # Return the prediction as JSON
    return jsonify({'prediction': str(prediction[0])})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000)
