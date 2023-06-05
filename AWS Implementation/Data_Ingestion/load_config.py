# Import modules
import configparser
import boto3

# Read AWS credentials from the config file
cfg_data = configparser.ConfigParser()
cfg_data.read('dl.cfg')   

# Save AWS credentials
access_key_id = cfg_data["AWS"]["access_key_id"]
secret_access_key = cfg_data["AWS"]["secret_access_key"]

# Save IAM role and IAM policy settings
RoleName = cfg_data["IAM"]["role_name"]
RoleARN = 'arn:aws:iam::341370630698:role/{}'.format(RoleName)
PolicyName = cfg_data["IAM"]["policy_name"]

account_id = boto3.client(
    'sts',
    aws_access_key_id = access_key_id,
    aws_secret_access_key = secret_access_key).get_caller_identity().get('Account')

# Kinesis Firehose
Region = cfg_data["Firehose"]["region"]
DeliveryStreamName = cfg_data["Firehose"]["stream_name"]
DeliveryStreamType = cfg_data["Firehose"]["delivery_stream_type"]
SizeInMBs = int(cfg_data["Firehose"]["size_in_mb"])
IntervalInSeconds = int(cfg_data["Firehose"]["interval_in_seconds"])
StreamARN = 'arn:aws:firehose:{}:{}:deliverystream/'.format(Region, account_id, DeliveryStreamName)

# S3 Bucket
Bucket = cfg_data['S3']['bucket_name']
BucketARN = 'arn:aws:s3:::{}'.format(Bucket)
