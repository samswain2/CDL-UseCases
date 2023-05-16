import boto3
import json

ec2 = boto3.client('ec2')
instance_id = 'your-instance-id'  # Replace with your instance ID

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    if 'source' in event and event['source'] == 'aws.events':
        # This is a CloudWatch Events event. Start the EC2 instance.
        start_ec2_instance()
    elif 'Records' in event and event['Records'][0]['eventSource'] == 'aws:s3' and event['Records'][0]['eventName'].startswith('ObjectCreated'):
        # This is an S3 event. Check if the created object's name starts with 'job_complete_'
        key = event['Records'][0]['s3']['object']['key']
        if key.startswith('job_complete_'):
            # The created object's name starts with 'job_complete_'. Stop the EC2 instance.
            stop_ec2_instance()

def start_ec2_instance():
    print("Starting EC2 instance")
    ec2.start_instances(InstanceIds=[instance_id])

def stop_ec2_instance():
    print("Stopping EC2 instance")
    ec2.stop_instances(InstanceIds=[instance_id])
