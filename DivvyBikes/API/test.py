import pandas as pd
import requests
import json

# Read the validation dataset
validation_data = pd.read_csv(r'DivvyBikes\Model\test_df.csv')

# Define columns to pass into api
api_columns = ["trips", "landmarks", "temp", "rel_humidity", "dewpoint", "apparent_temp", 
           "precip", "rain", "snow", "cloudcover", "windspeed",
           "60201", "60202", "60208", "60301", "60302", "60304", 
           "60601", "60602", "60603", "60604", "60605", "60606", 
           "60607", "60608", "60609", "60610", "60611", "60612", 
           "60613", "60614", "60615", "60616", "60617", "60618", 
           "60619", "60620", "60621", "60622", "60623", "60624", 
           "60625", "60626", "60628", "60629", "60630", "60632", 
           "60636", "60637", "60638", "60640", "60641", "60642", 
           "60643", "60644", "60645", "60646", "60647", "60649", 
           "60651", "60653", "60654", "60657", "60659", "60660", 
           "60661", "60696", "60804", 
           "hours_since_start", "Year sin", "Year cos", 
           "Week sin", "Week cos", "Day sin", "Day cos"]


validation_data = validation_data[api_columns]

# Define the API endpoint URL
api_url = "http://18.189.119.248:5000/predict"

# Num rows to test
num_test_rows = 1200

# Loop through the validation dataset and send requests to the API
results = []
for idx, row in validation_data.iloc[0:num_test_rows, :].iterrows():
    # Convert the row to a dictionary
    data = [row.to_dict()]
    
    # Send a POST request to the API with the data
    response = requests.post(api_url, json=data)
    # Parse the JSON response and store the prediction
    # prediction = json.loads(response.text)['prediction']
    # results.append(prediction)
    # print(data)
    # print(prediction)

# # Convert the results to a DataFrame
# results_df = pd.DataFrame(results, columns=['prediction'])

# # Save the results to a CSV file (optional)
# results_df.to_csv('api_results.csv', index=False)

# # Print the results
# print(results_df)
