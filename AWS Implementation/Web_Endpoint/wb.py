import websocket
import json
import threading
import time

API_GATEWAY_WEBSOCKET_URL = "wss://273h5twiyj.execute-api.us-east-2.amazonaws.com/production/"

def on_message(ws, message):
    data = json.loads(message)
    if data['action'] == 'send_prediction':
        print(f"Received prediction: {data['prediction']}")
    else:
        print(f"Unknown action: {data['action']}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("WebSocket connection closed")

def start_websocket():
    ws = websocket.WebSocketApp(
        API_GATEWAY_WEBSOCKET_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

websocket_thread = threading.Thread(target=start_websocket)
websocket_thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
