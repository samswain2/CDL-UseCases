import boto3
import os
import json

dynamodb = boto3.client('dynamodb')
CONNECTIONS_TABLE = 'websocket-connections-harddrive'

def lambda_handler(event, context):
    connection_id = event['requestContext'].get("connectionId")
    print(connection_id)
    try:
        dynamodb.put_item(
            TableName=CONNECTIONS_TABLE,
            Item={'connectionId': {'S': connection_id}}
        )
        print("Connection id sent to DynamoDB")
    except Exception as e:
        print(f"Error saving connection ID: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Failed to connect"})
        }
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Connected"})
    }