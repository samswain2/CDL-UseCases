import boto3

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')

    # specify the ID of the instance
    instance_id = 'i-0e6e2a4944f25a56c'

    # Start the instance
    response = ec2.start_instances(InstanceIds=[instance_id])

    return {
        'statusCode': 200,
        'body': response
    }