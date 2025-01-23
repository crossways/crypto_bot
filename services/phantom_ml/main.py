import requests
import pandas as pd
import json

symbol = "LINKUSDT"
target_interval = "1d"

req = requests.get(f"http://127.0.0.1:8000/candlesticks/{symbol}/{target_interval}")
stock_data = pd.DataFrame(json.loads(req.content.decode()))
