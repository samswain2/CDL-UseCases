#!/usr/bin/env python
# coding: utf-8

# # Ingest streaming data into S3 using Kinesis Firehose

# When interacting with AWS from a Jupyter notebook or python code, it is a good practise to store relevant data that allow to communicate with the cloud in a separate config file. In this tutorial, that file is called "dl.cfg" and is store in the same location as the current jupyter notebook. The file contains three sections:
# - AWS credentials (access key ID and secret access key) needed to programmatically access AWS
# - IAM role and IAM policy names
# - settings of the Kinesis Firehose delivery stream
# - S3 destination bucket name
# As a first step, let's extract some of the above mentioned parameters from "dl.cfg" file.

# In[1]:


import configparser
import time
import boto3
import json
import random

# Read AWS credentials from the config file
cfg_data = configparser.ConfigParser()
cfg_data.read('dl.cfg')   

# Save AWS credentials
access_key_id     = cfg_data["AWS"]["access_key_id"]
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


# The first step consists of creating a S3 bucket that will serve as a destination for the Kinesis Firehose delivery stream. This can be achieved through the following steps:
# - Define Boto3 S3 client to programmatically access S3
# - Define a function to create a new bucket
# - Run the function

# In[2]:


# Create S3 client feeding AWS credentials extracted from the config.json file
s3 = boto3.client(
    's3',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key)

# Define a function to create a S3 bucket
def create_bucket(Bucket):
    """
    Create a S3 bucket named 'Bucket'
    """
    
    # Check if a S3 bucket with the same name already exists
    try:
        s3.head_bucket(Bucket=Bucket)
        print('Bucket {} already exists'.format(Bucket))
    except:
        print('Bucket {} does not exist or you have no access'.format(Bucket))
        
        print('Creating bucket {}...'.format(Bucket))

        # Create a new bucket
        response = s3.create_bucket(Bucket=Bucket)
    
        # Loop until the bucket has been created succesfully
        created = False
        while not created:

            for bucket in s3.list_buckets()['Buckets']:
                if bucket['Name'] == Bucket:
                    created = True
                    break
        print('Bucket {} successfully created'.format(Bucket))
        return response
    
# Run the function defined above to create a new S3 bucket
create_bucket(Bucket)
    


# Check if a IAM Role with the same name already exists and delete it if it does. Then create a new IAM Role to enable Kinesis Firehose to write to S3

# In order to allow Kinesis Firehose writing data into a S3 bucket, an Identity Access Management (IAM) role should be created. This role will allow AWS services to be called on behalf of the user. Similarly to S3 service, AWS IAM service can be accessed by python SDK Boto3 using a specific client. In the code below, the following operations will be executed:
# - define client to control IAM
# - check if any role with the name defined in the config file already exists and (if it does) delete it
# - create a new role destined to Kinesis Firehose.

# In[3]:


# Create IAM client feeding AWS credentials extracted from the config.json file
iam = boto3.client(
    "iam",
    aws_access_key_id = access_key_id,
    aws_secret_access_key = secret_access_key
)

# Try to delete the existing role with the same name, if it exists
try:
    role = iam.get_role(RoleName = RoleName)
    
    print("Role named '{}' already exists".format(RoleName))

    # Extract all the attached policies to the existing role
    attached_policies = iam.list_attached_role_policies(RoleName = RoleName)["AttachedPolicies"]

    # Iterate over all attached policies
    for attached_policy in attached_policies:

        # Extract attached policy ARN
        attached_policy_arn = attached_policy["PolicyArn"]

        # Detach policy from role
        iam.detach_role_policy(
            RoleName = RoleName,
            PolicyArn = attached_policy_arn
        )

    # Delete role
    try:
        delete_role = iam.delete_role(RoleName = RoleName)
        print("Role named '{}' has been deleted".format(RoleName))

    except Exception as e:
        print(str(e))
        
except Exception as e:
    print(str(e))

# Create new IAM role
try:
    role = iam.create_role(
        RoleName = RoleName,
        Description = "Allows Kinesis Firehose Stream to write to S3",
        AssumeRolePolicyDocument = json.dumps(
            {
             "Version": "2012-10-17",
             "Statement": {
               "Effect": "Allow",
               "Principal": {"Service": "firehose.amazonaws.com"},
               "Action": "sts:AssumeRole"
              }
            } 

        )
    )
    print("Role '{}' has been created".format(RoleName))

except Exception as e:
    print(str(e))
 
# Extract role ARN
RoleARN = iam.get_role(RoleName=RoleName)["Role"]["Arn"]
print("Role '{}'s ARN is: '{}'".format(RoleName, RoleARN))


# An IAM role does not grant by default any permission to access specific AWS services. What determines which specific services are accessible is defined by an IAM policy. IAM policies are written in JSON and consist of a list of statements; each statement defines one or more actions, an effect (Allow or Deny), and a resource which the statement is applied to.
# In the code below, the following operations will be executed:
# - check if a policy with the name defined in the config file already exists
# - if a policy already exists, detach the policy from all the role it is attached to
# - delete all versions of the policy (including the default version)
# - create a new policy allowing Kinesis Firehose specific permissions for the destination S3 bucket
# - attach the policy to the role created above.

# In[4]:


# Check if policy with the wanted name already exists
try:
    policies = iam.list_policies()["Policies"]
    policy_exists = False
    for policy in policies:
        if policy["PolicyName"] == PolicyName:
            existing_policy_arn = policy["Arn"]
            policy_exists = True
            break          
except:
    print(str(e))

# If a policy with the same name already exists, delete it
if policy_exists:
    print("Policy named '{}' already exists".format(PolicyName))
    
    # Extract all roles
    roles = iam.list_roles()["Roles"]
    
    # Iterate over all the roles
    for role in roles:
        
        # Extract role name
        existing_role_name = role["RoleName"]
        
        # Extract all the attached policy to the role
        attached_policies = iam.list_attached_role_policies(
            RoleName = existing_role_name
        )["AttachedPolicies"]
        
        # Iterate over all the attached policies
        for attached_policy in attached_policies:

            # Extract attached policy ARN
            attached_policy_arn = attached_policy["PolicyArn"]

            # Checking if the policy correspond to the wanted one
            if attached_policy_arn == existing_policy_arn:
                
                # Detach policy from role
                iam.detach_role_policy(
                    RoleName = existing_role_name,
                    PolicyArn = attached_policy_arn
                )
                
                print("Policy with ARN '{}' detached from role '{}'".format(PolicyArn, existing_role_name))
    
    # Extract all the policy versions
    policy_versions = iam.list_policy_versions(
        PolicyArn = existing_policy_arn
    )["Versions"]
    
    # Iterate over all the policy versions
    for policy_version in policy_versions:
        
        # Skip the version if it is a default version
        if policy_version["IsDefaultVersion"]:
            continue
          
        # Extract policy ID
        version_id = policy_version["VersionId"]
        
        # Delete policy version
        iam.delete_policy_version(
            PolicyArn = existing_policy_arn,
            VersionId = version_id
        )
        print("Policy with ARN '{}', version_ID '{}' deleted".format(existing_policy_arn, version_id))
    
    # Delete default version of the policy
    iam.delete_policy(
        PolicyArn = existing_policy_arn
    )
    print("Policy with ARN '{}' deleted".format(existing_policy_arn))
    
else:
    print("Policy named '{}' does not exists".format(PolicyName))
 
PolicyContent = {
                "Version": "2012-10-17",  
                "Statement":
                [    
                    {      
                        "Effect": "Allow",      
                        "Action": [
                            "s3:AbortMultipartUpload",
                            "s3:GetBucketLocation",
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:ListBucketMultipartUploads",
                            "s3:PutObject"
                        ],      
                        "Resource": [        
                            "arn:aws:s3:::{}".format(Bucket),
                            "arn:aws:s3:::{}/*".format(Bucket)
                        ]    
                    },        
                    {
                        "Effect": "Allow",
                        "Action": [
                            "kinesis:DescribeStream",
                            "kinesis:GetShardIterator",
                            "kinesis:GetRecords",
                            "kinesis:ListShards"
                        ],
                        "Resource": "arn:aws:kinesis:{}:{}:stream/{}".format(Region, account_id, DeliveryStreamName)
                    },
                ]
            }

# Create policy 
try:
    policy = iam.create_policy(
        PolicyName = PolicyName,
        Description = "Allow to list and access content of the target bucket 'receive-streaming-data'",
        PolicyDocument = json.dumps(PolicyContent)        
    )
    print("Policy named '{}' created".format(PolicyName))
    PolicyArn = policy["Policy"]["Arn"]
    print("Policy named '{}' has ARN '{}'".format(PolicyName, PolicyArn))
except Exception as e:
    print(str(e))

# Attach policy to IAM role
try:
    attachment = iam.attach_role_policy(
        RoleName = RoleName,
        PolicyArn = PolicyArn
    )
    print("Policy named '{}' attached to role '{}'".format(PolicyName, RoleName))
except Exception as e:
    print(str(e))


# Using the boto3 client for Kinesis Firehose, the following functions are created:
# - a delete_stream function whose goal is to identify if a delivery stream with the same name exists and delete it
# - a create_stream function whose goal is to create a new delivery stream; for this tutorial, the stream operates with a "Put Records" delivery method and the buffer limits are setup as 5MB/s and 60 seconds a a GZIP compression is used. These parameters can be easily replaced in the "dl.cfg" file.

# In[5]:


# Create Kinesis Firehose client feeding AWS credentials extracted from the config.json file
firehose = boto3.client(
    'firehose',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key)

def delete_stream(DeliveryStreamName):
    """
    The function deletes an existing stream named 'DeliveryStreamName'
    """
    
    # Delete the current stream with the same name
    response = firehose.delete_delivery_stream(
        DeliveryStreamName=DeliveryStreamName,
        AllowForceDelete=True
    )

    # Get status of the stream 
    status = firehose.describe_delivery_stream(
    DeliveryStreamName=DeliveryStreamName)[
        'DeliveryStreamDescription']['DeliveryStreamStatus']
    print('{} stream "{}" ...'.format(status, DeliveryStreamName))

    # Wait until the stream is deleted
    i = 0
    while status == 'DELETING':
        time.sleep(10)
        print('Stream "{}" is being deleted, {} seconds elapsed...'.format(DeliveryStreamName, 30*(i+1)))
        try:
            status = firehose.describe_delivery_stream(
                DeliveryStreamName=DeliveryStreamName)['DeliveryStreamDescription']['DeliveryStreamStatus']
            i += 1
        except:
            status = 'DELETED'
    print('Stream "{}" has been succesfully deleted'.format(DeliveryStreamName))

    return status

def create_stream(
    DeliveryStreamName,
    RoleARN,
    BucketARN,
    SizeInMBs=SizeInMBs,
    IntervalInSeconds=IntervalInSeconds,
):
    """
    The function creates a new stream named 'DeliveryStreamName'
    """
         
    # Create a new stream
    response_create = firehose.create_delivery_stream(
        DeliveryStreamName=DeliveryStreamName,
        DeliveryStreamType='DirectPut',
        S3DestinationConfiguration={
            'RoleARN': RoleARN,
            'BucketARN': BucketARN,
            'BufferingHints': {
                'SizeInMBs': SizeInMBs,
                'IntervalInSeconds': IntervalInSeconds
            },
        },
    )
    

    # Get the status of the new stream
    status = firehose.describe_delivery_stream(
        DeliveryStreamName=DeliveryStreamName)['DeliveryStreamDescription']['DeliveryStreamStatus']
    print('{} stream "{}" ...'.format(status, DeliveryStreamName))

    # Wait until the stream is active
    i = 0
    while status == 'CREATING':
        time.sleep(10)
        print('Stream "{}" is being created, {} seconds elapsed...'.format(DeliveryStreamName, 30*(i+1)))
        status = firehose.describe_delivery_stream(
        DeliveryStreamName=DeliveryStreamName)['DeliveryStreamDescription']['DeliveryStreamStatus']
        i += 1

    # Check that the stream is active
    if status == 'ACTIVE':
        print('Stream "{}" has been succesfully created'.format(DeliveryStreamName))
        stream_arn = response_create['DeliveryStreamARN']
        print('Stream "{}" ARN: {}'.format(DeliveryStreamName, stream_arn))
    elif status == 'CREATING_FAILED':
        print('Stream "{}" creation has failed'.format(DeliveryStreamName))

    return status


# In the following code the two functions defined above are run in order to create a Kinesis Firehose delivery stream according to the parameters defined in the 'dl.cfg' file

# In[10]:


# Check if there is an existing stream with the same name in the same region

try:
    list_stream = firehose.list_delivery_streams()
    
    replace = 'yes'
    
    # Check if the stream already exists
    if DeliveryStreamName in list_stream['DeliveryStreamNames']:

        
        # Ask the user if the stream should be replaced
        replace = input("Stream {} already exists. Do you want to replace it? Type 'yes' to replace, otherwise 'no'".format(DeliveryStreamName))
        print(replace)
        
        # If the user has chosen to replace the stream, delete it and create a new one
        if replace == 'yes':
            
            # Delete stream
            try:
                status = delete_stream(DeliveryStreamName)
            
            except Exception as e:
                print(str(e))
            
            
            # Create new stream
            try:
                status = create_stream(
                    DeliveryStreamName=DeliveryStreamName,
                    RoleARN=RoleARN,
                    BucketARN=BucketARN,
                    SizeInMBs=SizeInMBs,
                    IntervalInSeconds=IntervalInSeconds)
            
            except Exception as e:
                print(str(e))            

        # If the user has chosen not to replace the stream, do nothing
        elif replace == 'no':

            None
            
        else:
            print('input not valid')
            
    # If the stream does not exist, proceed and create a new one
    else:
        
        try:
            status = create_stream(
                DeliveryStreamName=DeliveryStreamName,
                RoleARN=RoleARN,
                BucketARN=BucketARN,
                SizeInMBs=SizeInMBs,
                IntervalInSeconds=int(IntervalInSeconds))

        except Exception as e:
            print(str(e)) 

        
except Exception as e:
    print(str(e))
        


# After the delivery stream has been succesfully created, it can be tested by producing some sample records and streaming them to the delivery stream using the "Put Record" method

# In[20]:


# Define a sample record

# Send each record to the delivery stream
for i in range(10000):
    response = firehose.put_record(
        DeliveryStreamName=DeliveryStreamName,
        Record={
            'Data': json.dumps(
                {
                    "sensorId": random.randrange(1,3,1),
                    "currentTemperature": random.randrange(0,35,1),
                    "status": "OK"
                }
            )
        }
    )
    time.sleep(1)
    print(response)
    


# Finally, the delivery stream can be deleted to avoid extra cost.

# In[9]:


# Delete stream

# Ask the user if the stream should be deleted
delete = input("Do you want to delete Stream {}? Type 'yes' to delete, otherwise 'no'".format(DeliveryStreamName))

# If the user has chosen to replace the stream, delete it and create a new one
if delete == 'yes':

    # Delete stream
    try:
        status = delete_stream(DeliveryStreamName)

    except Exception as e:
        print(str(e))

