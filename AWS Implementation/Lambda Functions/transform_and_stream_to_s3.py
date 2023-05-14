import json
import pandas as pd
import boto3
from datetime import datetime
import base64


# import boto3

landmark  = pd.read_csv('s3://usecases-glue-jobs/divvy/static/landmark_clean.csv')
weather = pd.read_csv('s3://usecases-glue-jobs/divvy/streamed/weather_streamed.csv')

def join_dataframes(incoming_data, landmark, weather):
    # Perform the join operation(s) based on your requirements
    # Example: merge dataframes on a common column named 'id'
    result = incoming_data.merge(weather, left_on="start_time", right_on="time", how = "left").merge(landmark, left_on = "zip", right_on="zip_code", how='left')
    result = result.drop(columns = ['Unnamed: 0_x', 'Unnamed: 0_y', "Unnamed: 0"])
    return result

def lambda_handler(event, context):
    # Create an S3 client
    s3 = boto3.client('s3')

    # Iterate over each record
    for record in event['Records']:
        # Kinesis data is base64 encoded so decode here
        payload = base64.b64decode(record["kinesis"]["data"])

        # Assuming the payload is a json. If not, adjust this line accordingly
        json_payload = json.loads(payload)

        # Convert the incoming data point to a dataframe
        incoming_data = pd.DataFrame([json_payload])

        # Join the incoming data point with the two dataframes
        joined_data = join_dataframes(incoming_data, landmark, weather)

        # Define the S3 parameters
        s3params = {
            'Bucket': 'divvy-stream-data',
            'Key': 'kinesis_data/{}.json'.format(datetime.now().isoformat()),
            'Body': joined_data.to_json(orient='records')
        }

        # Write the data to S3
        try:
            s3.put_object(**s3params)
            print('Successfully wrote data to S3: {}'.format(s3params['Key']))
        except Exception as e:
            print('Error writing to S3: {}'.format(e))
