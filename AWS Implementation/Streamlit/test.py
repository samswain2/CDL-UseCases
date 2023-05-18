import websocket
import json

def on_open(ws):
    json_data = json.dumps({'data':'value'})
    ws.send(json_data)

def on_message(ws, message):
    print('data update: %s' % message)


apiUrl = "wss://zcu349obv5.execute-api.us-east-2.amazonaws.com/prod"
ws = websocket.WebSocketApp(apiUrl, on_message = on_message, on_open = on_open)
ws.run_forever()