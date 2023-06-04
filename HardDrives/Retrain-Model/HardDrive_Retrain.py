### ----------------- Import libraries ----------------- ###

# Data manipulation libs
import datetime
import logging
import joblib
import json
import tempfile
import os
import boto3
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

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

def filter_df_by_date(df, days_retrain):
    df["time_series_data"] = pd.to_datetime(df["date"])  # ensure that the column is in datetime format
    # Finding last date in data
    end_date = max(df_retrain['time_series_data'])

    # Finding date up to which retrain data can be collected
    start_date = end_date - datetime.timedelta(days=days_retrain)

    # Converting to string
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    mask = (df["time_series_data"] > start_date) & (df["time_series_data"] <= end_date)
    return df.loc[mask]

### ----------------- Settings ----------------- ###

# Initialize a boto3 client & buckets
s3 = boto3.client(
    's3'
)

stream_data_bucket = 'harddrive-stream-data'
retrain_bucket = 'harddrive-retraining'

train_path_key = "training-data/harddrive.csv"
# If you uncomment the line below you'll enable this script to read the new data it
# writes to an s3 bucket from a previous training job. Right now, the script is
# getting all streamed data. As of now if you enable that, you'll have duplicate rows
# so please impliment a funciton that moves the streamed data used for updating
# the dataframe into a different location before enabling 
# train_path_key = "training-data/updated_training_data.csv" # Uncommend when able

new_labels_path_key = "training-data/y_retrain_data_harddrive.csv"
streamed_data_prefix = "kinesis_data/"

ubuntu_path = "/home/ubuntu/"

# Define model columns
feature_columns = ["capacity_bytes",	"smart_1_normalized",\
    	"smart_1_raw",	"smart_3_normalized",	"smart_3_raw",\
        "smart_4_raw",	"smart_5_raw",	"smart_7_normalized",	"smart_9_normalized",\
    	"smart_9_raw",	"smart_12_raw",	"smart_194_normalized",	"smart_194_raw",\
        "smart_197_raw", "smart_199_raw"]
y_variable = ["useful_life"]

### ----------------- Import data ----------------- ### 

# Read current training data
df_train = read_csv_from_s3(retrain_bucket, train_path_key)
print("Old data shape: ", df_train.shape)
logging.info("Retrieved old data successfully")

# Get new training features
new_data = get_new_data_from_s3(stream_data_bucket, streamed_data_prefix)
new_data.drop(columns=[''], axis = 1, inplace=True)
print("New data shape: ", new_data.shape)
logging.info("Retrieved new data successfully")

# # Make sure that the new data and the labels have the same order
# new_data = new_data.sort_values('time_series_data')
# new_data = new_data[feature_columns]

# # Add the labels to the new data
# new_data['test_type'] = labels['test_type']

# Concat old and new training data
df_retrain = pd.concat([df_train, new_data], ignore_index=True)
print("Combined Data shape: ", df_retrain.shape)
logging.info("Combined Data succesfully.")

# # Save the dataframe to a csv
# csv_buffer = StringIO()
# df_retrain.to_csv(csv_buffer, index=False)

# # Put the CSV data to the S3 bucket
# s3.put_object(Bucket=retrain_bucket, Key='training-data/updated_training_data.csv', Body=csv_buffer.getvalue())
# logging.info("Saved updated dataframe to S3 bucket")
# logging.info('All data loaded')

##------------------ Select date based on date range -------------- ###
# It is recommended to have at least a quarter's (90 days) worth of data to retrain for this use case
# Filter the DataFrame by date range
logging.info("Starting to select data based on date range")
days_retrain = 90
df_retrain = filter_df_by_date(df_retrain, days_retrain)
logging.info("Data selected succesfully based on range.")

# ### ----------------- Transform data ----------------- ###
#TO model remaining useful life (based on Amram et al- Interpretable predictive maintenance for hard drives)
#Step 1 - Finding the failed hard drives
harddrive_failed = df_retrain.loc[df_retrain.failure==1]['serial_number']

df_analysis = df_retrain.loc[df_retrain.serial_number.isin(harddrive_failed)]
df_analysis["end_date"] = df_analysis.groupby("serial_number")['date'].transform("max")

df_analysis["end_date"] = pd.to_datetime(df_analysis["end_date"])
df_analysis["date"] = pd.to_datetime(df_analysis["date"])

df_analysis["useful_life"] = (df_analysis["end_date"] - df_analysis["date"])
print('Retraining data size: ', df_analysis.shape)
logging.info("Retraining data - y variable calculated successfully.")

col_to_drop = ['date','serial_number', 'model', 'end_date', 'failure', 'smart_5_normalized', 'smart_198_raw',
              'smart_198_normalized','smart_199_normalized','smart_241_raw','smart_240_raw','smart_10_raw',
               'smart_197_normalized','smart_188_raw','smart_12_normalized','smart_10_normalized','smart_7_raw','smart_4_normalized',
               'smart_242_raw', 'time_series_data']

needed_columns = feature_columns + y_variable

df_retrain_1 = df_analysis[needed_columns]
df_retrain_2 = df_retrain_1.apply(pd.to_numeric)
print("The size of transformed retrain dataset: ", df_retrain_2.shape)
print("The data types for transformed data: ")
print(df_retrain_2.dtypes)
logging.info("Final training data is set up.")

# ### ----------------- X and y definition ----------------- ###

X = df_retrain_2[feature_columns]
y = df_retrain_2["useful_life"]

# ### ----------------- Train and save model ----------------- ###

# Train the model
hyperparameter_dict = {'learning_rate': 0.1, 'max_depth': 8, 'max_leaves': 4, 'n_estimators': 200}
model = XGBRegressor(**hyperparameter_dict)
logging.info('Model created.')
model.fit(X, y)
logging.info('Model trained successfully.')

# Save the model
model_filename = "xgb_model_retrained.json"
model.save_model(model_filename)

# Upload to S3
upload_file_to_s3(retrain_bucket, 'training-artifacts/' + model_filename, ubuntu_path + model_filename)

# logging.info('Model saved and uploaded to S3')
