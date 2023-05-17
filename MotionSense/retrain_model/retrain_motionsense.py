### ----------------- Import libraries ----------------- ###

# Data manipulation libs
import logging
import joblib
import json
import tempfile
import os
import boto3
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Modeling
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

logging.info('Imports completed')

### ----------------- Settings ----------------- ###

# Define paths
train_path = "../CDL-UseCases/MotionSense/local_analysis/train_motionsense_lstm.csv"
new_data_path = "Streamed_Path"
new_labels_path = "../CDL-UseCases/MotionSense/local_analysis/y_retrain_data_motionsense.csv"

# Define model columns
feature_columns = [
    'attitude.roll', 'attitude.pitch', 'attitude.yaw',
    'gravity.x', 'gravity.y', 'gravity.z',
    'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
    'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'
    ]

### ----------------- Import data ----------------- ### 

# Read current training data
df_train = pd.read_csv(train_path)

# Get new training features

# Initialize a boto3 client
s3 = boto3.client(
    's3'
)

def get_new_data_from_s3(bucket_name, s3_prefix):

    # Initialize an empty DataFrame to store the new data
    new_data = pd.DataFrame()

    # List all files in the bucket with the specified prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

    # Check if the bucket contains any objects
    if 'Contents' not in response:
        return new_data

    # Loop through each file
    for file in response['Contents']:
        # Get the file name
        file_name = file['Key']

        # Check if the file is a JSON file
        if file_name.endswith('.json'):
            # Download the file to a temporary location
            tmp_file_path = os.path.join(tempfile.gettempdir(), 'tmp.json')
            s3.download_file(bucket_name, file_name, tmp_file_path)

            # Load the JSON file into a DataFrame
            df = pd.read_json(tmp_file_path, lines=True)

            # Append the DataFrame to the new data
            new_data = pd.concat([new_data, df], ignore_index=True)

    # Return the new data
    return new_data

# Define your bucket name and s3 prefix
bucket_name = "motionsense-stream-data"
s3_prefix = "kinesis_data/"

# Get new training features
new_data = get_new_data_from_s3(bucket_name, s3_prefix)

print(new_data)

logging.info("Retrieved new data successfully")

# Read labels from local file
labels = pd.read_csv(new_labels_path)

# Make sure that the new data and the labels have the same order
new_data = new_data.sort_values('time_series_data')
new_data = new_data[feature_columns]
# labels = labels.sort_values('timestamp')

# Add the labels to the new data
new_data['test_type'] = labels['test_type']

# Concat old and new training data
df_train = pd.concat([df_train, new_data], ignore_index=True)

df_train.to_csv("../CDL-UseCases/MotionSense/local_analysis/new_motionsense_training_data.csv")

logging.info("Saved updated dataframe")


# Get new training labels



# Concat old and new training data



logging.info('All data loaded')

### ----------------- Transform data ----------------- ###

### Scale features

# Scale features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df_train[feature_columns])
df_train[feature_columns] = scaler.transform(df_train[feature_columns])

logging.info('Data scaling completed')

# Save scaler
scaler_filename = "../CDL-UseCases/MotionSense/retrain_model/motionsense_lstm_scalar.save"
joblib.dump(scaler, scaler_filename)

logging.info('Scaler saved')

### Feature variables

# Set data/model attributes
n_timesteps = 50 # Set length of memory (# of observation model looks back)
n_categories = 6 # Set number of categories
n_features = len(feature_columns) # Set number of features
epochs = 1
batch_size = 64
optimizer = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# Convert df to 3D arrays
array_train_lstm = df_train[feature_columns].values

# Initialize arrays to store LSTM inputs
X_train_lstm = np.zeros((array_train_lstm.shape[0], n_timesteps, n_features))

# Loop through arrays for each set and create LSTM input
for i in range(n_timesteps, array_train_lstm.shape[0]):
    X_train_lstm[i-n_timesteps] = array_train_lstm[i-n_timesteps:i]

logging.info('X_train transformed successfully')

### Dependent variable

# Initilize encoder
encoder = LabelEncoder()

# Encode training y data and convert to categorical using one-hot encoding
encoder.fit(df_train["test_type"])
y_train_lstm = encoder.transform(df_train["test_type"])
y_train_lstm = to_categorical(y_train_lstm, num_classes = n_categories)

# Save label encoder
np.save("../CDL-UseCases/MotionSense/retrain_model/motionsense_lstm_label_encoder.npy", encoder.classes_)

logging.info("Label encoder saved successfully")

### Create/Train model ###

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    logging.info("GPU is available")
else:
    logging.info("No GPU found")

# Define model
def create_model():

    # Initialize a sequential model
    model = Sequential()

    # Add a bidirectional LSTM layer to the model
    model.add(Bidirectional(LSTM(units=16, input_shape=(n_timesteps, n_features))))

    # Add a dense output layer with 6 units and a softmax activation function
    model.add(Dense(n_categories, activation='softmax'))

    # Compile the model using the Adam optimizer, categorical crossentropy loss, and accuracy metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

model = create_model()

logging.info('Model created')

# Train model
model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=2)

logging.info('Model trained')

### ----------------- Save model ----------------- ###

# Save model
model_path = "../CDL-UseCases/MotionSense/retrain_model/"
model.save(model_path + "MotionSense_LSTM.h5")

logging.info('Model saved')