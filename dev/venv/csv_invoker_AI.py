#!/usr/bin/env python3
import os
import csv
import json
import re
import time
import requests
from pathlib import Path

try:
    from log_invoke import log_function_invoke
except ImportError:
    def log_function_invoke(func_name, payload):
        return {'functionStatus': 1}

AI_SERVICE_URL = "http://127.0.0.1:8000/predict"
INPUT_CSV = os.path.join(os.path.dirname(__file__), 'test_data2.csv')
LOG_FILE = './test_results_log.csv'
DELAY_BETWEEN_INVOCATIONS = 7000 / 1000  # Convert ms to seconds

def call_ai_service(func_name, user_id, status):
    """AI Prediction Service Call"""
    try:
        res = requests.post(AI_SERVICE_URL, json={
            'function': func_name,
            'user': str(user_id),
            'function_status': status,
            'timestamp': int(time.time() * 1000)
        }, timeout=3)
        return res.json()
    except Exception as e:
        return None

def log_detailed_result(user_id, func_name, status, ai_res):
    """Log results to the output CSV"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Activated_User', 'Function_Invoked', 'Function_Status', 
                           'Top1_Prediction', 'Top2_Prediction', 'Warmup_Function'])

    f_status = 'Success' if status == 1 else 'Failed'
    preds = (ai_res and ai_res.get('top2_predictions')) or []
    top1 = (preds[0].get('function') if preds and len(preds) > 0 else None) or 'N/A'
    top2 = (preds[1].get('function') if preds and len(preds) > 1 else None) or 'N/A'
    warmup = (ai_res and ai_res.get('warmup_function')) or 'None'

    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, func_name, f_status, top1, top2, warmup])

def run_csv_invoker():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        return

    # Read CSV lines and skip header
    with open(INPUT_CSV, 'r') as f:
        rows = [line.strip() for line in f.readlines() if line.strip()]
    
    rows = rows[1:]  # Skip header
    print(f"🚀 Starting Batch Process: {len(rows)} functions to invoke.")

    active_user = 'anonymous'
    session_counter = 0

    for i, row in enumerate(rows):
        # Parse CSV format: function_name,"{""key"":""value""}"
        match = re.match(r'^([^,]+),(.+)$', row)
        if not match:
            continue

        func_name = match.group(1).strip()
        payload_text = match.group(2).strip()
        
        # Remove outer quotes and unescape inner quotes
        if payload_text.startswith('"') and payload_text.endswith('"'):
            payload_text = payload_text[1:-1]
        payload_text = payload_text.replace('""', '"')
        
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {}

        # Logic to track user sessions based on login
        if 'login' in func_name:
            session_counter += 1
            active_user = payload.get('email') or f'session-{session_counter}'

        print(f"[{i + 1}/{len(rows)}] Invoking {func_name} for {active_user}...")

        # 1. Invoke the actual function via faasd
        invoke_result = log_function_invoke(func_name, payload)
        
        # 2. Get AI Prediction
        ai_res = call_ai_service(func_name, active_user, invoke_result.get('functionStatus', 1))

        # 3. Log results
        log_detailed_result(active_user, func_name, invoke_result.get('functionStatus', 1), ai_res)

        if i < len(rows) - 1:
            time.sleep(DELAY_BETWEEN_INVOCATIONS)

    print(f"✅ Batch complete. Results saved to {LOG_FILE}")

if __name__ == '__main__':
    run_csv_invoker()