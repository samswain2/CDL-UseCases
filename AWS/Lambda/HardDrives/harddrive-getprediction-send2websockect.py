import base64
import json
import boto3
import requests
from datetime import datetime

# Initialize the clients
dynamodb = boto3.client('dynamodb')
url = "https://dapybmt804.execute-api.us-east-2.amazonaws.com/production"
CONNECTIONS_TABLE = 'websocket-connections-harddrive'
api_gateway_management_api = boto3.client('apigatewaymanagementapi', endpoint_url=url)

def lambda_handler(event, context):
    # The URL of your API hosted on EC2
    api_url = "http://3.138.230.70:2000/predict"

    # Iterate over each record
    for record in event['Records']:
        # Kinesis data is base64 encoded so decode here
        payload = base64.b64decode(record["kinesis"]["data"])

        # Assuming the payload is a json. If not, adjust this line accordingly
        json_payload = [json.loads(payload)]

        # Make the request to your API
        response = requests.post(api_url, json=json_payload)

        # Check that the request was successful
        if response.status_code != 200:
            print("Error calling API: {} Response: {}".format(response.status_code, response.text))
            continue

        # Load the prediction from the API response
        prediction = response.json()
        
        prediction = [prediction["prediction"]]

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

