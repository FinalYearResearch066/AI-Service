import numpy as np
import requests
import time
import asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Any

app = FastAPI()

# --- HYPERPARAMETERS ---
ALPHA = 0.4
GAMMA = 0.3
EPSILON = 0.05
MAX_REPLICAS = 10
CPU_BINS = [20, 40, 60, 80, 100]

# --- FAASD & PROMETHEUS CONFIG ---
GATEWAY_URL = "http://127.0.0.1:8080"
PROM_URL = "http://127.0.0.1:9090/api/v1/query"
AUTH = ("admin", "tWgcKz29MgqQaRz0FMRutpsWsudI594PZE6YMCyJV6fXq0KWyeSfVAwW4EjZgty")

q_table = {}

def get_state_idx(replicas, cpu_util):
    cpu_bin_idx = next((i for i, v in enumerate(CPU_BINS) if cpu_util <= v), len(CPU_BINS) - 1)
    return (int(replicas) - 1, cpu_bin_idx)

def get_q_value(state, action):
    return q_table.get((state, action), 0.0)

def fetch_metrics(func_name):
    queries = {
        "cpu": f'sum(rate(pod_cpu_usage_seconds_total{{function_name="{func_name}"}}[1m])) * 100',
        "replicas": f'gateway_service_count{{function_name="{func_name}"}}',
        "success": f'sum(rate(gateway_function_invocation_total{{function_name="{func_name}",code="200"}}[5m]))',
        "failure": f'sum(rate(gateway_function_invocation_total{{function_name="{func_name}",code!="200"}}[5m]))'
    }
    results = {}
    for key, q in queries.items():
        try:
            r = requests.get(PROM_URL, params={'query': q}).json()
            results[key] = float(r['data']['result']['value'][1]) if r['data']['result'] else 0.0
        except:
            results[key] = 0.0
    return results

async def q_learning_agent(func_name):
    metrics = fetch_metrics(func_name)
    state = get_state_idx(max(1, metrics['replicas']), metrics['cpu'])
    
    if np.random.uniform(0, 1) < EPSILON:
        target_replicas = np.random.randint(1, MAX_REPLICAS + 1)
    else:
        actions = range(1, MAX_REPLICAS + 1)
        q_values = [get_q_value(state, a) for a in actions]
        target_replicas = actions[np.argmax(q_values)] if q_values else 1

    scale_url = f"{GATEWAY_URL}/system/scale-function/{func_name}"
    requests.post(scale_url, json={"service": func_name, "replicas": target_replicas}, auth=AUTH)

    await asyncio.sleep(300) 

    new_metrics = fetch_metrics(func_name)
    new_state = get_state_idx(max(1, target_replicas), new_metrics['cpu'])
    
    reward = (0.5 * new_metrics['cpu']) + (0.3 * (1/target_replicas)) + \
             (0.1 * new_metrics['success']) + (0.1 * new_metrics['failure'])
    
    if new_metrics['failure'] > 70 or new_metrics['cpu'] > 75:
        reward = -10

    old_q = get_q_value(state, target_replicas)
    actions = range(1, MAX_REPLICAS + 1)
    q_values_next = [get_q_value(new_state, a) for a in actions]
    max_next_q = max(q_values_next) if q_values_next else 0.0
    q_table[(state, target_replicas)] = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)

# --- FASTAPI ENDPOINTS ---
class Invocation(BaseModel):
    function_name: str
    input_data: Any  # CSV එකේ payload එක මෙතනින් භාරගනී

@app.post("/invoke")
async def handle_invocation(data: Invocation, background_tasks: BackgroundTasks):
    background_tasks.add_task(q_learning_agent, data.function_name)
    
    actual_func_url = f"{GATEWAY_URL}/function/{data.function_name}"
    # payload එක (input_data) සැබෑ function එක වෙත යොමු කරයි
    resp = requests.post(actual_func_url, json=data.input_data, auth=AUTH)
    
    return {
        "status": "request_forwarded", 
        "faasd_response": resp.status_code,
        "actual_output": resp.text # මෙතනින් සැබෑ function output එක පෙනේ
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)