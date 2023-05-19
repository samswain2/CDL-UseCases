import json
import os
import boto3

dynamodb = boto3.client('dynamodb')
url = "https://273h5twiyj.execute-api.us-east-2.amazonaws.com/production"
CONNECTIONS_TABLE = 'websocket-connections'
api_gateway_management_api = boto3.client('apigatewaymanagementapi', endpoint_url=url)


def lambda_handler(event, context):
    prediction = 69420
    response = dynamodb.scan(TableName=CONNECTIONS_TABLE)

    for item in response['Items']:
        connection_id = item['connectionId']['S']
        print(connection_id)
        try:
            api_gateway_management_api.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({
                    "action": "send_prediction",
                    "prediction": prediction
                })
            )
        except Exception as e:
            print(f"Error sending prediction to connection {connection_id}: {e}")
            # Remove the connection ID from the table if it's no longer valid
            if e.response['Error']['Code'] == 'GoneException':
                dynamodb.delete_item(
                    TableName=CONNECTIONS_TABLE,
                    Key={'connectionId': {'S': connection_id}}
                )

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Prediction sent to clients"})
    }





