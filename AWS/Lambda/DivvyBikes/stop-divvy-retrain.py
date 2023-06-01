import boto3

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')

    # specify the ID of the instance
    instance_id = 'i-0bc5b12e9bd2b429d'

    # Stop the instance
    response = ec2.stop_instances(InstanceIds=[instance_id])

    return {
        'statusCode': 200,
        'body': response
    }