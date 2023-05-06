import pandas as pd
import requests
import json
import os 

# Read the validation dataset
path = os.getcwd()
validation_data = pd.read_csv(path + "/test.csv")

# Define columns to pass into api
# Column names
column_names = ["capacity_bytes",	"smart_1_normalized",	"smart_1_raw",	"smart_3_normalized",	"smart_3_raw",	"smart_4_raw",	"smart_5_raw",	
                "smart_7_normalized",	"smart_9_normalized",	"smart_9_raw",	"smart_12_raw",	"smart_194_normalized",	"smart_194_raw",
                "smart_197_raw", "smart_199_raw", "useful_life"]

validation_data = validation_data[column_names]

# Define the API endpoint URL
api_url = "http://localhost:5000/predict"

# Num rows to test
num_test_rows = 100

# Loop through the validation dataset and send requests to the API
results = []
for idx, row in validation_data.iloc[0:num_test_rows, :].iterrows():
    # Convert the row to a dictionary
    data = [row.to_dict()]
    
    # Send a POST request to the API with the data
    response = requests.post(api_url, json=data)
    
    # Parse the JSON response and store the prediction
    prediction = json.loads(response.text)['prediction']
    results.append(prediction)
    print(prediction)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results, columns=['prediction'])

# Save the results to a CSV file (optional)
results_df.to_csv('api_results.csv', index=False)

# Print the results
print(results_df)
