import boto3
import io
import pandas as pd
import json
import streamlit as st
import time
import requests
from io import StringIO


def fetch_data_from_lambda(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        st.error(f"Error fetching data from Lambda: {response.status_code}")
        return None

def fetch_json_from_s3(bucket, prefix):
    profile_name = 'refit-iot'
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return objects

def json_to_pd(objects, bucket):
    combined_df = pd.DataFrame()
    profile_name = 'refit-iot'
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")
    for obj in objects['Contents']:
        file_key = obj['Key']
        file_obj = s3.get_object(Bucket=bucket, Key=file_key)
        file_content = file_obj['Body'].read().decode('utf-8')
        temp_df = pd.read_json(StringIO(file_content), lines=True)
        combined_df = pd.concat([combined_df, temp_df], axis = 0, ignore_index=True)

    return combined_df

def main():
    st.title("Streamlit test")

    st.header("Current prediction")
    api_url = "https://6k26mkkjd8.execute-api.us-east-2.amazonaws.com/motionsense-streamlit-temp"
    data = fetch_data_from_lambda(api_url)

    if data:
        st.write("Data from Lambda function:")
        st.write(data)

    if st.button('Retrieve past predictions'):
        json_list = fetch_json_from_s3('motionsense-predictions', 'prediction_data/')
        df = json_to_pd(json_list, 'motionsense-predictions')
        print(df)
        st.write(df)




    # container_1 = st.container()
    # col1,col2 = container_1.columns(2)
    # col3,col4 = st.columns(2)

    # with col1:
    #     cust_id = st.text_input("Enter Customer ID") ## textbox

    # ## Place holders
    # col3.write("Customer ID") 
    # col3.write("Gender")
    # col3.write("Internet Service")
    # col3.write("Contract")
    # col3.write("Payment Method")

    # if cust_id:

    #     ## Accessing the file in S3
    #     with st.spinner('Extracting the data from database'):
    #         session = boto3.Session(profile_name=profile_name)
    #         s3 = session.client("s3")
    #         obj = s3.get_object(Bucket=bucket_name, Key=test_file)
    #         data = obj['Body'].read()
    #         df_back = pd.read_excel(io.BytesIO(data))

    #     ## FIltering the values
    #     df_select = df_back[df_back["customerID"]==cust_id]

    #     check_pred = df_select['Probability'].values[0]
    #     check_pred = (check_pred) * 100
    #     check_pred = f'{check_pred:.2f}' + '%'

    #     col2.metric(label="Churn Probability", value= check_pred)
    #     col4.write(cust_id)
    #     col4.write(df_select['gender'].values[0])
    #     col4.write(df_select['InternetService'].values[0])
    #     col4.write(df_select['Contract'].values[0])
    #     col4.write(df_select['PaymentMethod'].values[0])





if __name__ == "__main__":
    main()


