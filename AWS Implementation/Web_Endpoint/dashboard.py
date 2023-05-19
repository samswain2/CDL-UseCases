import dash
from dash import dcc, html
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from collections import deque

import json
import websocket
import threading

current_prediction = None
prediction_buffer = deque(maxlen=10)
# Create a Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children='Real-time Predictions Dashboard'),
        html.Div(id='live-update-text'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # in milliseconds
            n_intervals=0
        )
    ]
)

# Global variable to store the prediction
current_prediction = None

def on_error(ws, error):
    print("Error occurred:", error)

def on_close(ws):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection opened")
    # You can send data here if needed
    # ws.send(json.dumps({"type": "start"}))

def on_message(ws, message):
    global prediction_buffer
    data = json.loads(message)
    prediction = data.get('prediction', None)
    print(prediction_buffer)
    if prediction is not None and (len(prediction_buffer) == 0 or prediction != prediction_buffer[-1]):
        prediction_buffer.append(prediction)
        print("Received data:", data)


def start_websocket():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "wss://273h5twiyj.execute-api.us-east-2.amazonaws.com/production", 
        on_open=on_open,
        on_close=on_close,
        on_message=on_message
    )
    ws.run_forever()

websocket_thread = threading.Thread(target=start_websocket, daemon=True)
websocket_thread.start()

@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    global prediction_buffer

    if len(prediction_buffer) == 0:
        raise PreventUpdate

    # Display all the predictions in the buffer
    predictions = ', '.join(map(str, prediction_buffer))
    return f'Latest predictions: {predictions}'

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
