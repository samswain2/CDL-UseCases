import json
import csv
import boto3
import time
from typing import List

def reading_csv(file_location: str) -> List:
	# Make JSON from the Hard Drive data CSV files
	HardDrive = []
	count = 0

	with open(file_location, encoding='utf-8') as csvf:
		csvReader = csv.DictReader(csvf)
		for rows in csvReader:
			HardDrive.append(rows)
			if count == 9:
				break
			count+=1

	return HardDrive

def create_kds(hard_drive: List) -> None:
	# Create a kinesis client
	region_name = 'us-east-2'
	client = boto3.client('kinesis', region_name = region_name)
	counter = 0

	for i in hard_drive:

		# Send message to Kinesis DataStream
		response = client.put_record(
			StreamName = "harddrive-stream",
			Data = json.dumps(i),
			PartitionKey = str(hash(i['serial_number']))
		)

		counter = counter + 1
		time.sleep(1)
		print('Message sent #' + str(counter))

		if response['ResponseMetadata']['HTTPStatusCode'] != 200:
			print('Error!')
			print(response)

if __name__ == '__main__':
	file_location = 'harddrive_stream.csv'
	file_contents = reading_csv(file_location)

	create_kds(file_contents)

