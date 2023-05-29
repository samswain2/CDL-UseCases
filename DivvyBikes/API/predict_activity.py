# Import libraries
import keras
import pandas as pd
from flask import Flask, request, jsonify
import json

from transform_data import WindowGenerator

# Create Flask app instance
app = Flask(__name__)

# Get model path and load model
model_path = "DivvyBikes_LSTM.h5"
model = keras.models.load_model(model_path)

# Column names
column_name = ["trips", "landmarks", "temp", "rel_humidity", "dewpoint", "apparent_temp", 
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

# Define variables
hif = 24
input_width = 30*24
dp_req = 30

# Buffer for incoming data
data_buffer = []

## Flask App ##

# Define the API endpoint and request method
@app.route("/predict", methods=["POST"])
def predict():
    global data_buffer  # use the global variable

    # Get the incoming data from the request
    data = request.get_json()
    #print(data)
    data_list = json.loads(data)
    data_dict = data_list[0]

    data1 = {col: data_dict[col] for col in column_name}

    # Convert the data into a DataFrame and add to the buffer
    data_buffer.append(pd.DataFrame(data1, columns=column_name, index=[0]))

    # If we have less than 30 data points, return a message
    if len(data_buffer) < dp_req:
        return jsonify({"message": f"Collecting data, {len(data_buffer)} data points collected so far."})

    # If we have 30 or more data points, generate a prediction
    if len(data_buffer) >= dp_req:
        sample = pd.concat(data_buffer[-dp_req:], ignore_index=True)
        w1 = WindowGenerator(input_width=input_width, label_width=hif, shift=hif, 
                             test_df=sample, label_columns=["trips"])
        
        print(w1)

        # Generate prediction
        prediction = model.predict(w1.test, verbose=False)
        print(prediction)

        # Return the prediction as JSON
        return jsonify({"Prediction": prediction.tolist()})


    # # data2 = {i: int(data1[i]) for i in list(data1.keys())}
    # # print(data2)
    # # Convert the data into a DataFrame
    # sample = pd.DataFrame.from_dict(data1, orient='index', columns=column_name)

    # # Transform the data and get the prediction
    # prediction = model.predict(sample)

    # # Return the prediction as JSON
    # return jsonify({'prediction': str(prediction[0])})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    