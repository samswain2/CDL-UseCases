from lambda_function import lambda_handler

# Define multiple test events
test_events = [
    {
        "data": {
            "attitude.roll": 0.01,
            "attitude.pitch": -0.23,
            "attitude.yaw": 0.21,
            "gravity.x": 0.29,
            "gravity.y": -0.91,
            "gravity.z": 0.10,
            "rotationRate.x": -0.12,
            "rotationRate.y": 0.02,
            "rotationRate.z": 0.09,
            "userAcceleration.x": 0.07,
            "userAcceleration.y": 0.07,
            "userAcceleration.z": 0.08
        }
    },
    # Add more test events here
    {
        "data": {
            "attitude.roll": 0.02,
            "attitude.pitch": -0.25,
            "attitude.yaw": 0.23,
            "gravity.x": 0.31,
            "gravity.y": -0.92,
            "gravity.z": 0.12,
            "rotationRate.x": -0.14,
            "rotationRate.y": 0.03,
            "rotationRate.z": 0.10,
            "userAcceleration.x": 0.09,
            "userAcceleration.y": 0.08,
            "userAcceleration.z": 0.09
        }
    }
]

# Test the lambda_handler function with each test event
for event in test_events:
    response = lambda_handler(event, None)
    print(response)
