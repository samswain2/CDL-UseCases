# CDL Industry Practicum Project: IoT Use Cases
This repository contains the implementation of three IoT use cases on AWS together with deliverables for the [practicum project](https://www.mccormick.northwestern.edu/analytics/curriculum/descriptions/msia-489.html) in collaboration with the Center for Deep Learning (CDL) at Northwestern University. 

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
The project was completed by the following MSiA students at Northwestern University: Yi (Betty) Chen, Henry Liang, Sharika Mahadevan, Ruben (Dias) Nakano, Riu Sakaguchi, Sam Swain, and Yumeng (Rena) Zhang. 

## IoT Use Cases

### Divvy Bikes
- Brief motivation of the problem at hand
- Briefly describe the modeling solution

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
- Hard Drives
- MotionSense

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

**Data Storage**

**Model Inference**

**Model Retraining**

**Predictions**

### Solution Cost Estimation

## Final Remarks
- Mention changes to project scope
- Mention final report for more details on the entire project. 