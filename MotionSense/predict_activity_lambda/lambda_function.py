import json
import pandas as pd
import numpy as np
from predict_activity import model, encoder, scaler, normalize_columns, transform_data, data_buffer, n_timesteps, n_features

def lambda_handler(event, context):
    # Check if event contains data
    if 'data' not in event:
        return {'statusCode': 400, 'body': 'Invalid input'}

    # Extract data from the event
    raw_data = event['data']

    # Convert the data to a DataFrame
    incoming_sample = pd.DataFrame([raw_data], columns=normalize_columns)

    # Transform the sample and update the buffer
    X = transform_data(incoming_sample, scaler, data_buffer, n_timesteps, n_features)

    # Make predictions with your LSTM model
    predictions = model.predict(X, verbose=False)

    # Find the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)

    # Convert the predicted class back to the original category
    predicted_category = encoder.inverse_transform(predicted_class)

    return {
        'statusCode': 200,
        'body': json.dumps({'predicted_category': predicted_category[0]})
    }
