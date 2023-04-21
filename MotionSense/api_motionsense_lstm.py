# Import libraries
import time
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import keras

# # Get model path and load model
model_path = './MotionSense/MotionSense_LSTM.h5'
model = keras.models.load_model(model_path)

# Load data
training_data = pd.read_csv('./MotionSense/train_motionsense_lstm.csv')

# Initilize encoder and set categories
encoder = LabelEncoder()
n_categories = 6

# Fit encoder
encoder.fit(training_data['test_type'])

# Define normalization function
scaler = MinMaxScaler(feature_range=(-1, 1))

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

# Fit the scaler to the training data
scaler.fit(training_data[normalize_columns])

# Store the number of features and number of timesteps back
n_features = len(training_data[normalize_columns].columns)
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





## Test function ##

# Load data
validation_data = pd.read_csv('./MotionSense/validation_motionsense_lstm.csv')

# Num rows to test
num_test_rows = 100

# Start the timer
start_time = time.time()

# Loop through rows in the validation data
for i in range(num_test_rows):
    # Incoming sample as a DataFrame
    incoming_sample = validation_data[normalize_columns].iloc[[i], :]

    # Transform the sample and update the buffer
    X = transform_data(incoming_sample, scaler, data_buffer, n_timesteps, n_features)

    # Make predictions with your LSTM model
    predictions = model.predict(X, verbose=False)

    # Find the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)

    # Convert the predicted class back to the original category
    predicted_category = encoder.inverse_transform(predicted_class)

    print(f"Sample {i + 1}:")
    print(f"Predicted Category: {predicted_category[0]}")
    print(f"Actual Category: {validation_data.loc[i, 'test_type']}")
    print("\n")

# Calculate the time taken
elapsed_time = time.time() - start_time

print(f"Time taken: {elapsed_time} seconds")