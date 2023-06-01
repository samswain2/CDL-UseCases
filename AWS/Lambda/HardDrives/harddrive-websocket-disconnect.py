import boto3
import os
import json

dynamodb = boto3.client('dynamodb')
CONNECTIONS_TABLE = 'websocket-connections-harddrive'

def lambda_handler(event, context):
    connection_id = event['requestContext'].get("connectionId")
    try:
        dynamodb.delete_item(
            TableName=CONNECTIONS_TABLE,
            Key={'connectionId': {'S': connection_id}}
        )
    except Exception as e:
        print(f"Error deleting connection ID: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Failed to disconnect"})
        }

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Disconnected"})
    }
