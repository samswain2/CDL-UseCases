# CDL Industry Practicum Project: IoT Use Cases
This repository contains the implementation of three IoT use cases on AWS together with deliverables for the [practicum project](https://www.mccormick.northwestern.edu/analytics/curriculum/descriptions/msia-489.html) in collaboration with the Center for Deep Learning (CDL) at Northwestern University. 

![](misc/flowchart.png)

## Overview

### Center for Deep Learning
The Center for Deep Learning’s mission is to act as a resource for companies seeking to establish or improve access to artificial intelligence (AI) by providing technical capacity and expertise. Their recent work include serving for deep learning, model architecture redesign, AI for IoT and general streaming, and prediction or scoring confidence. Please refer to the following [resource](https://www.mccormick.northwestern.edu/research/deep-learning/) for more information regarding CDL. 

### REFIT
The Center for Deep Learning is developing [REFIT](https://www.mccormick.northwestern.edu/research/deep-learning/projects/refit/), a novel system that is built to consume and capitalize on IoT infrastructure by ingesting device data and employing modern machine learning approaches to infer the status of various components of the IoT system. It is specifically built upon several open source components with state-of-the-art artificial intelligence and it is notably distinguished from other IoT systems in many regards. 

### Project Objectives

1. Develop and implement three IoT use cases based on public data.
2. Build and end-to-end solution for each use case on AWS, mimicking the general architecuture leveraged in REFIT. 
3. Assess the potential pros and cons of implementing a streaming-based solution in AWS versus REFIT. 

### Deliverables
1.  A comprehensive final report detailing the three IoT use cases, the end-to-end solution implemented in AWS, and a preliminary comparison between AWS and REFIT.
2. Source code and thorough documentation as provided in this GitHub repository.

### Contacts
- Point of Contact - Borchuluun Yadamsuren
- Technical Adviser - Diego Klabjan
- Supporting Staff - Raman Khurana

### Credits
The project was completed by the following MSiA students at Northwestern University: Yi (Betty) Chen, Henry Liang, Sharika Mahadevan, Ruben Nakano, Riu Sakaguchi, Sam Swain, and Yumeng (Rena) Zhang. 

## IoT Use Cases

### Divvy Bikes
A Chicago-based bike share system, Divvy Bikes provides an affordable and convenient mode of transportation throughout cities. The raw dataset provided publicly by Divvy contains information at the trip level, including the starting and ending station and time. The business objective revolves around predicting the number of trips at various stations for the next hour to facilitate resourceful restocking of bikes. The Divvy Bikes use case leverages an LSTM model to account for long-term seasonal dependencies to predict demand.

### Hard Drives
- Brief motivation of the problem at hand
- Briefly describe the modeling solution

### MotionSense
- Brief motivation of the problem at hand
- Briefly describe the modeling solution

## AWS Implementation

### Solution Architecture

<img width="613" alt="image" src="https://github.com/samswain2/CDL-UseCases/assets/113140351/df7f1658-4759-43fe-8698-952d9eda7fb3">

**Data Sources**
- [Divvy Bikes](https://divvybikes.com/system-data)
- [Hard Drives](https://www.backblaze.com/b2/hard-drive-test-data.html)
- [MotionSense](https://github.com/mmalekzadeh/motion-sense)

**Data Ingestion**
- Kinesis Data Streams
    - `divvy-stream`
    - `harddrive-stream`
    - `motionsense-stream`

- Amazon S3
    - `divvy-stream-data`
    - `harddrive-stream-data`
    - `motionsense-stream-data`

**Data Preparation**
- AWS Glue
    - `divvy_static_etl`

- AWS Lambda


**Data Storage**
- Amazon S3
    - `divvy-retraining`
    - `harddrive-retraining`
    - `motionsense-retraining`

- AWS Lambda

**Model Inference**
- AWS Lambda
- Amazon EC2
    - `divvy_api`
    - `harddrive_api`
    - `motionsense_api`
- Amazon S3

**Predictions**
- Amazon S3
    - `divvy-predictions`
    - `harddrive-predictions`
    - `motionsense-predictions`
- Amazon EventBridge
- Amazon SQS
- AWS Lambda
- WebSocket
    - `websocket-1`

**Model Retraining**
- Amazon EventBridge
- AWS Lambda
    - `trigger-motionsense-retrain`
- Amazon EC2
    - `motionsense_retrain`

### Solution Cost Estimation
The combined cost of the end-to-end AWS solution for the three use cases is estimated reach an **annual total of $3,207.22 USD** or equivalently, **$267.27 USD per month.** [Amazon API Gateway](https://aws.amazon.com/api-gateway/) and [AWS Glue](https://aws.amazon.com/glue/) are two of the more costly AWS services employed as part of the comprehensive solution. A detailed break down of the cost estimate by service can be found [here.](https://calculator.aws/#/estimate?id=1af19374e566120b06e9ec56d5f4bd66c7c329d3)

## Final Remarks
The final scope and objectives of the project has transitioned slightly from the original proposal including the implementation of the three use cases on REFIT and designing a model agnostic feature selection algorithm for time series data. These works could serve as potential avenues for consideration for future projects with CDL. 

Finally, the ***final report*** detailing the entire 8 month project can be found in `/Deliverables` directory. 
