import websocket
import json
import threading

def on_message(ws, message):
    data = json.loads(message)
    print("Received data:", data)

def on_error(ws, error):
    print("Error occurred:", error)

def on_close(ws):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection opened")
    # You can send data here if needed
    # ws.send(json.dumps({"type": "start"}))

def start_websocket():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "wss://273h5twiyj.execute-api.us-east-2.amazonaws.com/production", 
        on_open=on_open, 
        on_message=on_message, 
        on_error=on_error, 
        on_close=on_close
    )
    ws.run_forever()

websocket_thread = threading.Thread(target=start_websocket)
websocket_thread.start()

