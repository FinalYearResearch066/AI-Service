import requests
import json
import time
from datetime import datetime

# --- CONFIGURATION ---
SCALER_URL = "http://127.0.0.1:5000/invoke"
CSV_FILE = "test_data2.csv" 
REQUEST_INTERVAL = 30
CYCLE_DELAY = 7000

def load_requests(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header: return rows
        for line_no, raw_line in enumerate(f, start=2):
            line = raw_line.strip()
            if not line: continue
            parts = line.split(",", 1)
            if len(parts) < 2: continue
            func_name = parts[0].strip()
            payload_text = parts[1].strip()
            if payload_text.startswith('"') and payload_text.endswith('"'):
                payload_text = payload_text[1:-1].replace('""', '"')
            try:
                payload = json.loads(payload_text)
                rows.append({"function": func_name, "payload": payload})
            except: continue
    return rows

def run_csv_test():
    while True:
        try:
            requests_data = load_requests(CSV_FILE)
            print(f"\n--- Cycle Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            for index, row in enumerate(requests_data):
                func_name = row['function']
                payload = row['payload']
                
                request_data = {
                    "function_name": func_name,
                    "input_data": payload
                }

                print(f"[{index + 1}/200] {datetime.now().strftime('%H:%M:%S')} | Invoking: {func_name}...")
                
                try:
                    response = requests.post(SCALER_URL, json=request_data, timeout=60)
                    if response.status_code == 200:
                        res_json = response.json()
                        # මෙහිදී Scaler එකෙන් එවන 'actual_output' print කරයි
                        print(f"    📩 Faasd Status: {res_json.get('faasd_response')}")
                        print(f"    📊 Real Output: {res_json.get('actual_output')}")
                    else:
                        print(f"    ❌ Error: Scaler status {response.status_code}")
                except Exception as e:
                    print(f"    ⚠️ Error: {e}")

                if index < len(requests_data) - 1:
                    time.sleep(REQUEST_INTERVAL)

            print(f"\n--- Cycle complete. Next in 4 hours ---")
            time.sleep(CYCLE_DELAY)
        except Exception as e:
            print(f"Fatal error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_csv_test()