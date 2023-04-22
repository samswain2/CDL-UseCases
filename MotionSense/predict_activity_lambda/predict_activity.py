# Import libraries
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
import keras
import joblib

# # Get model path and load model
model_path = 'LSTM.h5'
model = keras.models.load_model(model_path)

# Load encoder
encoder = LabelEncoder()
encoder_filename = "label_encoder.npy"
encoder.classes_ = np.load(encoder_filename, allow_pickle=True)

# Load scalar
scaler_filename = "scalar.save"
scaler = joblib.load(scaler_filename) 

# Set num categories
n_categories = 6

# Define columns to normalize
normalize_columns = [
    'attitude.roll', 
    'attitude.pitch', 
    'attitude.yaw',
    'gravity.x', 
    'gravity.y', 
    'gravity.z',
    'rotationRate.x', 
    'rotationRate.y', 
    'rotationRate.z',
    'userAcceleration.x', 
    'userAcceleration.y', 
    'userAcceleration.z',
    # 'attitude', 
    # 'gravity', 
    # 'rotationRate', 
    # 'userAcceleration',
    # 'weight', 
    # 'height', 
    # 'age'
]

# Store the number of features and number of timesteps back
n_features = len(normalize_columns)
n_timesteps = 50

# Buffer to store the most recent samples
data_buffer = np.zeros((n_timesteps, n_features))

def transform_data(sample, scaler, data_buffer, n_timesteps, n_features):
    # Normalize the sample while keeping it as a DataFrame
    normalized_sample = pd.DataFrame(scaler.transform(sample), columns=sample.columns)
    
    # Update the data buffer
    data_buffer[:-1] = data_buffer[1:]
    data_buffer[-1] = normalized_sample.values
    
    # Reshape the buffer to be compatible with the LSTM model
    X = data_buffer.reshape(1, n_timesteps, n_features)
    
    return X