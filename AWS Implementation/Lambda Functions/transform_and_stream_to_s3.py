import json
import pandas as pd
import boto3
import base64
from datetime import datetime

# import boto3

landmark  = pd.read_csv('s3://usecases-glue-jobs/divvy/static/landmark_clean.csv')
weather = pd.read_csv('s3://usecases-glue-jobs/divvy/streamed/weather_streamed.csv')
ohe_zipcode = pd.read_csv('s3://usecases-glue-jobs/divvy/static/ohe_zipcode.csv')

def join_dataframes(incoming_data, landmark, weather):
    # Change the data types of columns to be the same before performing the merge
    incoming_data['start_time'] = pd.to_datetime(incoming_data['start_time'])
    weather['time'] = pd.to_datetime(weather['time'])
    incoming_data['zip'] = incoming_data['zip'].astype(str)
    ohe_zipcode['zip'] = ohe_zipcode['zip'].astype(str)
    landmark['zip_code'] = landmark['zip_code'].astype(str)
    
    # Perform the join operation(s) based on your requirements
    # Example: merge dataframes on a common column named 'id'
    result = incoming_data.merge(weather, left_on="start_time", right_on="time", how="left").merge(landmark, left_on="zip", right_on="zip_code", how='left')
    result = result.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])
    result = result.merge(ohe_zipcode, left_on="zip", right_on = "zip", how = 'left')
    result = result.drop(columns = ['zip', "time", "Unnamed: 0"])
    
    return result


def lambda_handler(event, context):
    # Create an S3 client

    s3 = boto3.client('s3')

    # d = pd.read_csv('s3://usecases-glue-jobs/divvy/streamed/streamed.csv')
    # joined_data = join_dataframes(d, landmark, weather)
    # print(joined_data.head(5))


    # Iterate over each record
    for record in event['Records']:
        # Kinesis data is base64 encoded so decode here
        print("raw", record["kinesis"]["data"])
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