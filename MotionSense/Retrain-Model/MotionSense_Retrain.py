### ----------------- Import libraries ----------------- ###

# Data manipulation libs
import logging
import joblib
import json
import tempfile
import os
import boto3
from io import StringIO
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

### ----------------- Functions ----------------- ###

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

            # Move the file to another location
            new_key = file_name.replace(s3_prefix, 'kinesis_data_old/')
            move_s3_file(bucket_name, file_name, new_key)

    # Return the new data
    return new_data

def download_file_from_s3(bucket_name, file_key, tmp_file_path):
    s3.download_file(bucket_name, file_key, tmp_file_path)

def upload_file_to_s3(bucket_name, file_key, file_path):
    s3.upload_file(file_path, bucket_name, file_key)

def read_csv_from_s3(bucket_name, file_key):
    tmp_file_path = os.path.join(tempfile.gettempdir(), 'tmp.csv')
    download_file_from_s3(bucket_name, file_key, tmp_file_path)
    return pd.read_csv(tmp_file_path)

def filter_df_by_date(df, start_date, end_date):
    df["time_series_data"] = pd.to_datetime(df["time_series_data"])  # ensure that the column is in datetime format
    mask = (df["time_series_data"] > start_date) & (df["time_series_data"] <= end_date)
    return df.loc[mask]

def save_df_to_s3(df, bucket_name, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())

def load_df_from_s3(bucket_name, key):
    tmp_file_path = os.path.join(tempfile.gettempdir(), 'tmp.csv')
    download_file_from_s3(bucket_name, key, tmp_file_path)
    return pd.read_csv(tmp_file_path)

def move_s3_file(bucket_name, original_key, new_key):
    # Copy the file to the new location
    s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': original_key}, Key=new_key)

    # Delete the file from the original location
    s3.delete_object(Bucket=bucket_name, Key=original_key)

### ----------------- Settings ----------------- ###

# Initialize a boto3 client & buckets
s3 = boto3.client(
    's3'
)

stream_data_bucket = 'motionsense-stream-data'
retrain_bucket = 'motionsense-retraining'

train_path_key = "training-data/train_motionsense_lstm.csv"
# If you uncomment the line below you'll enable this script to read the new data it
# writes to an s3 bucket from a previous training job. Right now, the script is
# getting all streamed data. As of now if you enable that, you'll have duplicate rows
# so please implement a function that moves the streamed data used for updating
# the dataframe into a different location before enabling 
# train_path_key = "training-data/updated_training_data.csv" # Uncommend when able

new_labels_path_key = "training-data/y_retrain_data_motionsense.csv"
streamed_data_prefix = "kinesis_data/"

ubuntu_home_path = "/home/ubuntu/"

# Define model columns
feature_columns = [
    'attitude.roll', 'attitude.pitch', 'attitude.yaw',
    'gravity.x', 'gravity.y', 'gravity.z',
    'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
    'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'
    ]

### ----------------- Import data ----------------- ### 

# Read current training data
# Load data. Uncomment the appropriate line to load either the full data or the windowed data
df_train = load_df_from_s3(retrain_bucket, train_path_key)  # load full data
# df_train = load_df_from_s3(retrain_bucket, 'training-data/windowed_training_data.csv')  # load windowed data

logging.info("Retrieved old data successfully")

# Get new training features
new_data = get_new_data_from_s3(stream_data_bucket, streamed_data_prefix)

if new_data.empty:
    logging.info("No new data available.")
else:
    # Make sure that the new data and the labels have the same order
    new_data = new_data.sort_values('time_series_data')
    new_data = new_data[feature_columns]

    logging.info("Retrieved new data successfully")

    # Read labels from local file
    labels = read_csv_from_s3(retrain_bucket, new_labels_path_key)

    logging.info("Retrieved new labels successfully")

    # Add the labels to the new data
    new_data['test_type'] = labels['test_type']

    # Concat old and new training data
    df_train = pd.concat([df_train, new_data], ignore_index=True)

# Filter the DataFrame by date range
start_date = '2022-01-01'
end_date = '2022-01-02'
df_train = filter_df_by_date(df_train, start_date, end_date)

# Save the dataframe to a csv
csv_buffer = StringIO()
df_train.to_csv(csv_buffer, index=False)

logging.info(df_train["test_type"].unique())
nan_rows = df_train[df_train['test_type'].isna()]
logging.info(len(nan_rows))

# Save the windowed DataFrame to S3
save_df_to_s3(df_train, retrain_bucket, 'training-data/windowed_training_data.csv')

logging.info("Saved updated dataframe to S3 bucket")

logging.info('All data loaded')

### ----------------- Transform data ----------------- ###

### Scale features

# Scale features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df_train[feature_columns])
df_train[feature_columns] = scaler.transform(df_train[feature_columns])

logging.info('Data scaling completed')

# Save scaler
scaler_filename = "motionsense_lstm_scalar.save"
joblib.dump(scaler, ubuntu_home_path + scaler_filename)

# Upload scaler to S3
upload_file_to_s3(retrain_bucket, 'training-artifacts/' + scaler_filename, ubuntu_home_path + scaler_filename)

logging.info('Scaler saved and uploaded to S3')

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
encoder_filename = "motionsense_lstm_label_encoder.npy"
np.save(ubuntu_home_path + encoder_filename, encoder.classes_)

# Upload label encoder to S3
upload_file_to_s3(retrain_bucket, 'training-artifacts/' + encoder_filename, ubuntu_home_path + encoder_filename)

logging.info("Label encoder saved and uploaded to S3")

### ----------------- Create/Train/Save model ----------------- ###

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
model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=1)

logging.info('Model trained')

# Save model
model_filename = "MotionSense_LSTM.h5"
model.save(ubuntu_home_path + model_filename)

# Upload model to S3
upload_file_to_s3(retrain_bucket, 'training-artifacts/' + model_filename, ubuntu_home_path + model_filename)

logging.info('Model saved and uploaded to S3')