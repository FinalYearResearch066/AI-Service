import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import uvicorn
import os
import csv
import json
from collections import Counter
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical

# TensorFlow configuration
tf.config.run_functions_eagerly(True)

app = FastAPI()

# ========================================================
# CONFIGURATION
# ========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "gru_dataset_fixed_dates_8000.csv")
MODEL_PATH = os.path.join(BASE_DIR, "gru_status_model.h5")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

ONLINE_LEARN_INTERVAL = 10
GATEWAY_URL = "http://127.0.0.1:8080"
PATTERN_MAX_SUFFIX_LEN = 3

# ========================================================
# MODEL ASSETS
# ========================================================
class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        return super().get_config()

# Load ML Assets
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"SimpleAttention": SimpleAttention})
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

func_enc = encoders["f"]
user_enc = encoders["u"]
time_enc = encoders["t"]
SEQUENCE_LEN = int(model.input_shape[0][1])

class InvocationData(BaseModel):
    function: str
    user: str
    timestamp: float
    function_status: int = 1
    functionStatus: int | None = None

# ========================================================
# HELPERS & PREPROCESSING
# ========================================================

def normalize_function_name(raw_function: str) -> str:
    func_name = str(raw_function).strip()
    if func_name in func_enc.classes_: return func_name
    path_tail = func_name.rsplit("/", 1)[-1].strip()
    if path_tail in func_enc.classes_: return path_tail
    raise ValueError(f"Invalid function value '{raw_function}'")

def encode_hour_series_to_time_idx(hour_series: pd.Series) -> np.ndarray:
    normalized_hour = hour_series.astype(str).str.strip().str.replace('.', ':', regex=False)
    hour_dt = pd.to_datetime(normalized_hour, format='%I:%M:%S %p', errors='coerce')
    
    labels = hour_dt.dt.strftime('%H:%M:%S')
    classes = np.asarray(time_enc.classes_, dtype=str)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    encoded = labels.map(label_to_idx)
    
    if encoded.notna().all():
        return encoded.astype(np.int32).to_numpy()

    # Fallback to nearest time
    input_seconds = (hour_dt.dt.hour * 3600 + hour_dt.dt.minute * 60 + hour_dt.dt.second).to_numpy(dtype=np.int32)
    known_dt = pd.to_datetime(classes, format='%H:%M:%S')
    known_seconds = (known_dt.hour * 3600 + known_dt.minute * 60 + known_dt.second).to_numpy(dtype=np.int32)
    insert_pos = np.searchsorted(known_seconds, input_seconds)
    nearest = np.where(insert_pos >= len(known_seconds), len(known_seconds)-1, insert_pos)
    return nearest.astype(np.int32)

# ========================================================
# HYBRID LOGIC: AI + PATTERN SCORING
# ========================================================

def choose_warmup_function_from_pattern(user_df: pd.DataFrame):
    if len(user_df) < 2:
        return None, {'matched_suffix_len': 0}

    user_events = list(zip(user_df['function'].astype(str).tolist(), 
                           pd.to_numeric(user_df['status'], errors='coerce').fillna(0).astype(int).tolist()))
    
    max_len = min(PATTERN_MAX_SUFFIX_LEN, len(user_events) - 1)
    
    for suffix_len in range(max_len, 0, -1):
        suffix = tuple(user_events[-suffix_len:])
        next_functions = []
        for idx in range(len(user_events) - suffix_len):
            if tuple(user_events[idx:idx + suffix_len]) == suffix:
                next_functions.append(user_events[idx + suffix_len][0])
        
        if next_functions:
            selected = Counter(next_functions).most_common(1)[0][0]
            return selected, {'matched_suffix_len': suffix_len}
            
    return user_events[-1][0], {'matched_suffix_len': 0}

def get_hybrid_decision(ai_func, ai_conf, pattern_func, pattern_len):
    """
    පර්යේෂණයේ හරය: AI Confidence සහ Pattern Reliability සසඳා තීරණය ගැනීම.
    """
    # Pattern එකේ ශක්තිය අනුව ලකුණු (Scoring)
    pattern_score_map = {3: 0.95, 2: 0.85, 1: 0.45, 0: 0.0}
    pattern_score = pattern_score_map.get(pattern_len, 0.0)

    # 1. AI සහ Pattern එකම දේ පවසයි නම් (Agreement)
    if ai_func == pattern_func and (ai_conf > 0.4 or pattern_score > 0.4):
        return [ai_func], f"Strong Agreement (AI:{ai_conf:.2f}, PatScore:{pattern_score})"

    # 2. Pattern එක වඩාත් ශක්තිමත් නම්
    if pattern_score > ai_conf:
        return [pattern_func], f"Pattern Dominant (PatScore:{pattern_score} > AI:{ai_conf:.2f})"

    # 3. AI එක වඩාත් විශ්වාසදායක නම්
    if ai_conf > pattern_score and ai_conf > 0.5:
        return [ai_func], f"AI Dominant (AI:{ai_conf:.2f} > PatScore:{pattern_score})"

    # 4. දෙකම දුර්වල නම්
    return [], "Low Confidence - No Warmup"

# ========================================================
# ONLINE LEARNING & BACKGROUND TASKS
# ========================================================

def append_to_csv(data: InvocationData):
    dt = datetime.fromtimestamp(data.timestamp / 1000.0)
    day, hour = dt.strftime('%#m/%#d/%Y'), dt.strftime('%#I:%M:%S %p')
    status = data.functionStatus if data.functionStatus is not None else data.function_status
    with open(CSV_PATH, mode='a', newline='') as f:
        csv.writer(f).writerow([data.function, data.user, day, hour, data.timestamp, data.timestamp, int(status)])

def trigger_warmup(func_name: str):
    try:
        requests.post(f"{GATEWAY_URL}/function/{func_name}", json={}, timeout=0.5)
        print(f"🔥 [Warm-up] Triggered: {func_name}")
    except: pass

# ========================================================
# MAIN PREDICT ENDPOINT
# ========================================================

@app.post("/predict")
async def predict(data: InvocationData, background_tasks: BackgroundTasks):
    try:
        data.function = normalize_function_name(data.function)
        append_to_csv(data)
        
        df = pd.read_csv(CSV_PATH)
        user_df = df[df['user'].astype(str) == str(data.user)].copy()
        
        if len(user_df) < SEQUENCE_LEN:
            return {"status": "Need more data", "warmup_triggered": False}

        # 1. AI Prediction Logic
        recent_hist = user_df.tail(SEQUENCE_LEN)
        f_in = func_enc.transform(recent_hist['function'].astype(str)).reshape(1, SEQUENCE_LEN)
        u_in = np.array([[user_enc.transform([str(data.user)])[0]] * SEQUENCE_LEN])
        t_in = encode_hour_series_to_time_idx(recent_hist['hour']).reshape(1, SEQUENCE_LEN)
        s_in = pd.to_numeric(recent_hist['status']).fillna(0).clip(0,1).values.reshape(1, SEQUENCE_LEN)

        with tf.device('/CPU:0'):
            pred_probs = model.predict([f_in, u_in, t_in, s_in], verbose=0)[0]
        
        ai_idx = np.argmax(pred_probs)
        ai_conf = float(np.max(pred_probs))
        ai_func = func_enc.inverse_transform([ai_idx])[0]

        # 2. Pattern Match Logic
        pattern_func, pat_details = choose_warmup_function_from_pattern(user_df)
        pat_len = pat_details['matched_suffix_len']

        # 3. Hybrid Decision Logic (මෙතැනදී AI vs Pattern සංසන්දනය වේ)
        warmup_targets, reason = get_hybrid_decision(ai_func, ai_conf, pattern_func, pat_len)

        # 4. Execution
        for target in warmup_targets:
            background_tasks.add_task(trigger_warmup, target)

        return {
            "prediction_ai": ai_func,
            "ai_confidence": ai_conf,
            "pattern_match": pattern_func,
            "pattern_len": pat_len,
            "final_decision": warmup_targets,
            "reason": reason,
            "warmup_triggered": bool(warmup_targets)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)