import base64
import json
import boto3
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize the clients
dynamodb = boto3.client('dynamodb')
url = "https://4q1o50coq0.execute-api.us-east-2.amazonaws.com/production"
CONNECTIONS_TABLE = 'websocket-connections-divvybikes'
api_gateway_management_api = boto3.client('apigatewaymanagementapi', endpoint_url=url)

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
    result = result.sort_values('start_time').reset_index(drop = True)
    
    start = result['start_time'][0]
    result['hours_since_start'] = (result['start_time'] - start).dt.total_seconds()/3600
    result = result.drop(labels='start_time', axis = 1)

    # Add yearly, weekly and daily signals 
    result['Year sin'] = np.sin(result['hours_since_start'] * (2 * np.pi / (365*24)))
    result['Year cos'] = np.cos(result['hours_since_start'] * (2 * np.pi / (365*24)))
    result['Week sin'] = np.sin(result['hours_since_start'] * (2 * np.pi / (7*24)))
    result['Week cos'] = np.cos(result['hours_since_start'] * (2 * np.pi / (7*24)))
    result['Day sin'] = np.sin(result['hours_since_start'] * (2 * np.pi / (24)))
    result['Day cos'] = np.cos(result['hours_since_start'] * (2 * np.pi / (24)))
    return result

def lambda_handler(event, context):
    global incoming_buffer
    # The URL of your API hosted on EC2
    api_url = "http://18.189.119.248:5000/predict"

    # Iterate over each record
    for record in event['Records']:
        # Kinesis data is base64 encoded so decode here
        payload = base64.b64decode(record["kinesis"]["data"])

        # Assuming the payload is a json. If not, adjust this line accordingly
        json_payload = [json.loads(payload)]
        
        # Convert the incoming data point to a dataframe
        incoming_data = pd.DataFrame([json_payload][0])

        # Join the incoming data point with the two dataframes
        joined_data = join_dataframes(incoming_data, landmark, weather)
        joined_json = joined_data.to_json(orient='records')

        # Make the request to your API
        response = requests.post(api_url, json=joined_json)

        # Check that the request was successful
        if response.status_code != 200:
            print("Error calling API: {} Response: {}".format(response.status_code, response.text))
            continue

        # Load the prediction from the API response
        
        prediction = response.json()
        
        try:
            prediction = [prediction["Prediction"]]
        except:
            print("Data buffer loading.")
            continue
        
        response = dynamodb.scan(TableName=CONNECTIONS_TABLE)
        for p in prediction:
            for item in response['Items']:
                connection_id = item['connectionId']['S']
                try:
                    api_gateway_management_api.post_to_connection(
                        ConnectionId=connection_id,
                        Data=json.dumps({
                            "action": "send_prediction",
                            "prediction": p
                        })
                    )
                    print("pred sent")
                except Exception as e:
                    print(f"Error sending prediction to connection {connection_id}: {e}")
                    # Remove the connection ID from the table if it's no longer valid
                    if 'Error' in e.response and 'Code' in e.response['Error'] and e.response['Error']['Code'] == 'GoneException':
                        dynamodb.delete_item(
                            TableName=CONNECTIONS_TABLE,
                            Key={'connectionId': {'S': connection_id}}
                        )

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Prediction sent to clients"})
    }
