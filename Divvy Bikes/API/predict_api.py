# Import libraries
import joblib
import keras
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

# Create Flask app instance
app = Flask(__name__)

# Get model path and load model
model_path = "../Model/Divvy_LSTM.h5"
model = keras.models.load_model(model_path)

## Flask App ##

# Define the API endpoint and request method
@app.route("/predict", methods=["POST"])
def predict():
    # Get the incoming data from the request
    data = request.get_json()

    # Convert the data into a DataFrame
    sample = pd.DataFrame(data, columns=normalize_columns)

    # Transform the data and get the prediction
    X = transform_data(sample, scaler, data_buffer, n_timesteps, n_features)
    prediction = model.predict(X, verbose=False)

    # Get the class label for the prediction
    #class_label = encoder.inverse_transform(prediction.argmax(axis=-1))[0]

    # Return the prediction as JSON
    return jsonify({"prediction": class_label})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)