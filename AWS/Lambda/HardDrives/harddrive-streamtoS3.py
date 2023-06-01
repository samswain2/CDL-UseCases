import boto3
import base64
import json
from datetime import datetime

def lambda_handler(event, context):
    # Create an S3 client
    s3 = boto3.client('s3')

    # Iterate over each record
    for record in event['Records']:
        # Kinesis data is base64 encoded so decode here
        payload = base64.b64decode(record["kinesis"]["data"])
        
        # Assuming the payload is a json. If not, adjust this line accordingly
        json_payload = json.loads(payload)

        # Define the S3 parameters
        s3params = {
            'Bucket': 'harddrive-stream-data',
            'Key': 'kinesis_data/{}.json'.format(datetime.now().isoformat()),
            'Body': json.dumps(json_payload)
        }

        # Write the data to S3
        try:
            s3.put_object(**s3params)
            print('Successfully wrote data to S3: {}'.format(s3params['Key']))
        except Exception as e:
            print('Error writing to S3: {}'.format(e))
