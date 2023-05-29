import boto3
import pandas as pd
from io import StringIO
import json

def lambda_handler(event, context):
    s3 = boto3.client('s3')

    # Read the original training data from S3
    response = s3.get_object(Bucket='motionsense-training-data', Key='train_motionsense_lstm.csv')
    original_training_data = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

    # Read the labels data from S3
    response = s3.get_object(Bucket='motionsense-training-data', Key='y_val_motionsense_lstm.csv')
    labels_data = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

    # List and read new data files from S3
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket='motionsense-stream-data', Prefix='kinesis_data')

    new_data_frames = []

    for page in pages:
        for obj in page['Contents']:
            response = s3.get_object(Bucket='motionsense-stream-data', Key=obj['Key'])
            file_content = response['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)
            df = pd.json_normalize(json_content)
            new_data_frames.append(df)

    new_data = pd.concat(new_data_frames, ignore_index=True)

    # Concatenate original training data with new data
    total_training_data = pd.concat([original_training_data, new_data], ignore_index=True)

    # Join the labels data
    # Assuming 'id' is the column on which to join. Replace 'id' with the actual column name(s).
    total_training_data = total_training_data.join(labels_data.set_index('id'), on='id')

    # Write the final DataFrame to a CSV file and store it back to S3
    csv_buffer = StringIO()
    total_training_data.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket='motionsense-training-data', Key='new_train_motionsense_lstm', Body=csv_buffer.getvalue())

    return {
        'statusCode': 200,
        'body': json.dumps('Data processing completed successfully!')
    }
