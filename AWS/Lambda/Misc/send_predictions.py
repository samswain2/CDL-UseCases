import json
import boto3

def lambda_handler(event, context):
    
    url = "https://273h5twiyj.execute-api.us-east-2.amazonaws.com/production"
    client = boto3.client("apigatewaymanagementapi", endpoint_url = url)
    
    pred = {"predicted value":69420} 
    
    response = client.post_to_connection(ConnectionId = event['requestContext'].get("connectionId"), Data=json.dumps(pred))
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
