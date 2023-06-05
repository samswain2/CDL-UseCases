### ----------------- Import libraries ----------------- ###

# Data manipulation libs
import logging
import json
import tempfile
import os
import boto3
from io import StringIO
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modeling
import tensorflow as tf

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logging.info('Imports completed')

### ----------------- Functions ----------------- ###

class S3Handler:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def get_new_data_from_s3(self, bucket_name, s3_prefix):
        new_data = pd.DataFrame()
        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

        if 'Contents' not in response:
            return new_data

        for file in response['Contents']:
            file_name = file['Key']
            if file_name.endswith('.json'):
                tmp_file_path = os.path.join(tempfile.gettempdir(), 'tmp.json')
                self.s3.download_file(bucket_name, file_name, tmp_file_path)

                with open(tmp_file_path) as f:
                    data = json.load(f)

                df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
                
                # Convert start_time from Unix timestamp in milliseconds to datetime
                df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
                
                # Convert datetime to string format 'YYYY-MM-DD HH:MM:SS'
                df['start_time'] = df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                new_data = pd.concat([new_data, df], ignore_index=True)

        return new_data

    def download_file_from_s3(self, bucket_name, file_key, tmp_file_path):
        self.s3.download_file(bucket_name, file_key, tmp_file_path)

    def upload_file_to_s3(self, bucket_name, file_key, file_path):
        self.s3.upload_file(file_path, bucket_name, file_key)

    def read_csv_from_s3(self, bucket_name, file_key):
        tmp_file_path = os.path.join(tempfile.gettempdir(), 'tmp.csv')
        self.download_file_from_s3(bucket_name, file_key, tmp_file_path)
        return pd.read_csv(tmp_file_path)

def create_windowed_df(df, start_date, end_date):
    mask = (df["start_time"] > start_date) & (df["start_time"] <= end_date)
    windowed_df = df.loc[mask]
    return windowed_df

def move_data_s3(bucket_name, old_prefix, new_prefix):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=old_prefix):
        old_source = {'Bucket': bucket_name, 'Key': obj.key}
        new_key = obj.key.replace(old_prefix, new_prefix, 1)
        new_obj = bucket.Object(new_key)
        new_obj.copy(old_source)
        obj.delete()

def load_data(filename="", bucket_name="", s3_key=""):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key + filename)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    return df

### ----------------- Settings ----------------- ###

handler = S3Handler()

stream_data_bucket = "divvy-stream-data"
retrain_bucket = "divvy-retraining"
train_path_key = "train_df.csv"
full_train_path_key = "training-data/full_data.csv"
window_train_path_key = "training-data/windowed_data.csv"
streamed_data_prefix = "kinesis_data/"
old_streamed_data_prefix = "kinesis_data_old/"
ubuntu_home_path = "/home/ubuntu/"

logging.info("Set Keys")

feature_columns = ["trips", "landmarks", "temp", 
                   "rel_humidity", "dewpoint", "apparent_temp", 
                   "precip", "rain", "snow", "cloudcover", "windspeed", 
                   "60201", "60202", "60208", "60301", "60302", "60304", 
                   "60601", "60602", "60603", "60604", "60605", "60606", 
                   "60607", "60608", "60609", "60610", "60611", "60612", 
                   "60613", "60614", "60615", "60616", "60617", "60618", 
                   "60619", "60620", "60621", "60622", "60623", "60624", 
                   "60625", "60626", "60628", "60629", "60630", "60632", 
                   "60636", "60637", "60638", "60640", "60641", "60642", 
                   "60643", "60644", "60645", "60646", "60647", "60649", 
                   "60651", "60653", "60654", "60657", "60659", "60660", 
                   "60661", "60696", "60804", 
                   "hours_since_start", "Year sin", "Year cos", 
                   "Week sin", "Week cos", "Day sin", "Day cos"]

### ----------------- Import data ----------------- ###

# Use load_data function to read current training data
df_train = load_data(filename=train_path_key, bucket_name=retrain_bucket)

logging.info("Retrieved old data successfully")

# Get new training features
new_data = handler.get_new_data_from_s3(stream_data_bucket, streamed_data_prefix)

logging.info("Retrieved new data successfully")

# After retrieving new data, move it to a new location
move_data_s3(stream_data_bucket, streamed_data_prefix, old_streamed_data_prefix)

logging.info("Moved processed data to new location")

# Make sure that the new data and the labels have the same order
new_data = new_data.sort_values("start_time")

logging.info("new_data shape: %s", new_data.shape)

# Concat old and new training data
df_train = pd.concat([df_train, new_data], ignore_index=True)

# Save the full old+new DataFrame to a CSV file and upload to S3
csv_buffer = StringIO()
df_train.to_csv(csv_buffer, index=False)
handler.s3.put_object(Bucket=retrain_bucket, Key=full_train_path_key, Body=csv_buffer.getvalue())

logging.info("Saved full old+new DataFrame to S3 bucket")

# Create a windowed dataframe with a given start and end date
start_date = "2017-04-09"
end_date = "2018-04-09"
df_train_windowed = create_windowed_df(df_train, start_date, end_date)

# Save the windowed DataFrame to a CSV file and upload to S3
csv_buffer = StringIO()
df_train_windowed.to_csv(csv_buffer, index=False)
handler.s3.put_object(Bucket=retrain_bucket, Key=window_train_path_key, Body=csv_buffer.getvalue())

logging.info("Saved windowed DataFrame to S3 bucket")

nan_rows = df_train[df_train["trips"].isna()]
logging.info("nan_rows shape: %s", nan_rows.shape[0])

logging.info("Saved updated dataframe to S3 bucket")

logging.info("All data loaded")

# ### ----------------- Transform data ----------------- ###

# Comment below to use entire dataframe for training
df_train = df_train_windowed

df_train = df_train[feature_columns]
logging.info("df_train shape: %s", df_train.shape[0])

# Get the column indices
column_indices = {name: i for i, name in enumerate(df_train.columns)}

num_features = df_train.shape[1]

# logging.info(df_train.head())

### Create windows

class WindowGenerator():
    '''
    - Input width: length of given history (i.e. length in time of training data)
    - Shift: make prediction n units of time in the future
    - Label width: # of predictions made in the future

    total_window_size, input_indicies, label_indices are attributes of WindowGenerator object
    '''

    def __init__(self, input_width, label_width, shift,
                train_df=df_train, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = train_df
        self.test_df = train_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

def plot(self, model=None, plot_col='trip_count', max_subplots=3, plot_title=None):
    ''' 
    Plots inputs, labels, predictions
    '''
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        if n == 0 and plot_title is not None:
            plt.title(plot_title)
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()
    
    plt.xlabel('Day')

WindowGenerator.plot = plot

def split_window(self, features):
    '''
    Converts total window into a window of inputs and a window of labels
    '''
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

WindowGenerator.split_window = split_window

def make_dataset(self, data):
    ''' 
    Takes time series dataset and turns it into tf.data.Dataset of (input_window, label_window)
    '''
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=256,)

    ds = ds.map(self.split_window)

    return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train

# takes 30 days of data, forecasts next 24 hours.
hours_into_future = 24
w1 = WindowGenerator(input_width=30*24, label_width=hours_into_future, shift=hours_into_future,
                     label_columns=['trips'], train_df=df_train)

### ----------------- Create/Train/Save model ----------------- ###

class TensorFlowModel:
    def __init__(self):
        self.gpu_check()
        self.model = self.build_model()
        
    @staticmethod
    def gpu_check():
        if tf.config.list_physical_devices('GPU'):
            logging.info("GPU is available")
        else:
            logging.info("No GPU found")

    @staticmethod
    def build_model(hours_into_future=1, num_features=1):
        # Define model
        OUT_STEPS = hours_into_future

        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
            # Shape => [batch, out_steps, features].
        ])

        logging.info('Model created')

        return multi_lstm_model

    @staticmethod
    def compile_and_fit(model, window, patience=2, MAX_EPOCHS=1):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                            patience=patience,
                                                            mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS, callbacks=[early_stopping], verbose=1)
        
        return history

    def train_and_save(self, window, model_filename, s3_handler, retrain_bucket):
        # Train model
        history = self.compile_and_fit(self.model, window)

        logging.info('Model trained')

        # Save model
        self.model.save(model_filename)

        # Upload model to S3
        s3_handler.upload_file_to_s3(retrain_bucket, 'training-artifacts/' + model_filename, model_filename)

        logging.info('Model saved and uploaded to S3')

model = TensorFlowModel()
model.train_and_save(w1, ubuntu_home_path + "DivvyBikes_LSTM.h5", handler, retrain_bucket)
