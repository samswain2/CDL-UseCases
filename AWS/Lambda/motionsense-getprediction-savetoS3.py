import base64
import json
import boto3
import requests
from datetime import datetime

def lambda_handler(event, context):
    # Create an S3 client
    s3 = boto3.client('s3')

    # The URL of your API hosted on EC2
    api_url = "http://13.58.130.45:5000/predict"
    

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

        # Define the S3 parameters
        s3params = {
            'Bucket': 'motionsense-predictions',
            'Key': 'prediction_data/{}.json'.format(datetime.now().isoformat()),
            'Body': json.dumps(prediction)
        }

        # Write the prediction to S3
        try:
            s3.put_object(**s3params)
            print('Successfully wrote prediction to S3: {}'.format(s3params['Key']))
        except Exception as e:
            print('Error writing prediction to S3: {}'.format(e))