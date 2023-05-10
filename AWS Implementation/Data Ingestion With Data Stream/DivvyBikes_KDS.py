import json
import csv
import boto3
from typing import List

def reading_csv(file_location: str) -> List:
	# Make JSON from the Divvy trip data CSV files
	DivvyRides = []
	count = 0

	with open(file_location, encoding='utf-8') as csvf:
		csvReader = csv.DictReader(csvf)
		for rows in csvReader:
			DivvyRides.append(rows)
			if count == 1050:
				break
			count+=1

	return DivvyRides

def create_kds(divvy_rides: List) -> None:
	# Create a kinesis client
	client = boto3.client('kinesis')
	counter = 0

	for ride in divvy_rides:

		# Send message to Kinesis DataStream
		response = client.put_record(
			StreamName = "divvy_example",
			Data = json.dumps(ride),
			PartitionKey = str(hash(ride['trip_id']))
		)

		counter = counter + 1

		# print('Message sent #' + str(counter))

		if response['ResponseMetadata']['HTTPStatusCode'] != 200:
			print('Error!')
			print(response)

if __name__ == '__main__':
	file_location = 'data/trips_full.csv'
	file_contents = reading_csv(file_location)

	create_kds(file_contents)
