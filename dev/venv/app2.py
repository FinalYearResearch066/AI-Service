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

# TensorFlow configuration to avoid "numpy()" error
tf.config.run_functions_eagerly(True)

app = FastAPI()

# ========================================================
# CONFIGURATION
# ========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..","gru_dataset_fixed_dates_8000.csv")
MODEL_PATH = os.path.join(BASE_DIR, "gru_status_model2.h5")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders2.pkl")

ONLINE_LEARN_INTERVAL = 10
GATEWAY_URL = "http://127.0.0.1:8082"
PATTERN_MAX_SUFFIX_LEN = 3
INFREQUENT_FUNCTION_THRESHOLD = 50


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


def normalize_function_name(raw_function: str) -> str:
    func_name = str(raw_function).strip()

    if func_name in func_enc.classes_:
        return func_name

    try:
        obj = json.loads(func_name)
        if isinstance(obj, dict):
            for key in ("function", "function_name", "functionName", "name"):
                candidate = str(obj.get(key, "")).strip()
                if candidate in func_enc.classes_:
                    return candidate
    except Exception:
        pass

    # Allow path-like values such as "/function/login".
    path_tail = func_name.rsplit("/", 1)[-1].strip()
    if path_tail in func_enc.classes_:
        return path_tail

    raise ValueError(
        f"Invalid function value '{raw_function}'. Send a valid function name from {list(func_enc.classes_)}."
    )


def encode_hour_series_to_time_idx(hour_series: pd.Series) -> np.ndarray:
    normalized_hour = (
        hour_series
        .astype(str)
        .str.strip()
        .str.replace('.', ':', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
    )
    hour_dt = pd.to_datetime(normalized_hour, format='%I:%M:%S %p', errors='coerce')
    if hour_dt.isna().any():
        bad_values = normalized_hour[hour_dt.isna()].head(5).tolist()
        raise ValueError(f"Unparseable 'hour' values found, examples: {bad_values}")

    labels = hour_dt.dt.strftime('%H:%M:%S')
    classes = np.asarray(time_enc.classes_, dtype=str)

    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    encoded = labels.map(label_to_idx)
    if encoded.notna().all():
        return encoded.astype(np.int32).to_numpy()

    # Fallback for unseen timestamps: map each unseen time to nearest known class.
    input_seconds = (
        hour_dt.dt.hour * 3600
        + hour_dt.dt.minute * 60
        + hour_dt.dt.second
    ).to_numpy(dtype=np.int32)
    known_dt = pd.to_datetime(classes, format='%H:%M:%S')
    known_seconds = (
        known_dt.hour * 3600
        + known_dt.minute * 60
        + known_dt.second
    ).to_numpy(dtype=np.int32)

    insert_pos = np.searchsorted(known_seconds, input_seconds)
    left_idx = np.clip(insert_pos - 1, 0, len(known_seconds) - 1)
    right_idx = np.clip(insert_pos, 0, len(known_seconds) - 1)

    left_dist = np.abs(input_seconds - known_seconds[left_idx])
    right_dist = np.abs(known_seconds[right_idx] - input_seconds)
    nearest = np.where(right_dist < left_dist, right_idx, left_idx)
    return nearest.astype(np.int32)

#heirarchy of warmup target selection logic:
def get_top_k_predictions(prob_vector: np.ndarray, encoder, k: int = 2):
    top_k = min(k, prob_vector.shape[0])
    top_indices = np.argsort(prob_vector)[-top_k:][::-1]
    return [
        {
            "function": str(encoder.inverse_transform([int(idx)])[0]),
            "confidence": float(prob_vector[int(idx)]),
            "value": float(prob_vector[int(idx)]),
        }
        for idx in top_indices
    ]


def is_infrequent_function(function_name: str, function_counts: pd.Series) -> bool:
    return int(function_counts.get(function_name, 0)) <= INFREQUENT_FUNCTION_THRESHOLD


def get_warmup_targets(
    top2_functions: list[str],
    pattern_warmup_function: str | None,
    function_counts: pd.Series,
) -> tuple[list[str], str]:
    if any(is_infrequent_function(function_name, function_counts) for function_name in top2_functions):
        warmup_targets = list(dict.fromkeys(top2_functions))
        return warmup_targets, "infrequent_top_prediction"

    if pattern_warmup_function is None:
        return [], "no_warmup_target"

    return [pattern_warmup_function], "pattern_match"


def get_sorted_user_history(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    user_history = df[df['user'].astype(str) == user_id].copy()

    if 't0' in user_history.columns:
        return user_history.sort_values('t0', kind='stable')
    if 'timestamp' in user_history.columns:
        return user_history.sort_values('timestamp', kind='stable')

    return user_history


def get_current_user_pattern(user_history: pd.DataFrame, max_suffix_len: int = PATTERN_MAX_SUFFIX_LEN) -> list[str]:
    user_functions = user_history['function'].astype(str).tolist()
    if not user_functions:
        return []
    return user_functions[-min(len(user_functions), max_suffix_len):]


def get_current_user_event_pattern(user_history: pd.DataFrame, max_suffix_len: int = PATTERN_MAX_SUFFIX_LEN) -> list[tuple[str, int]]:
    normalized_status = (
        pd.to_numeric(user_history['status'], errors='coerce')
        .fillna(0)
        .clip(0, 1)
        .astype(np.int32)
        .tolist()
    )
    user_events = list(zip(user_history['function'].astype(str).tolist(), normalized_status))
    return user_events[-min(len(user_events), max_suffix_len):]


def choose_warmup_function_from_pattern(user_history: pd.DataFrame, max_suffix_len: int = PATTERN_MAX_SUFFIX_LEN):
    normalized_status = (
        pd.to_numeric(user_history['status'], errors='coerce')
        .fillna(0)
        .clip(0, 1)
        .astype(np.int32)
        .tolist()
    )
    user_events = list(zip(user_history['function'].astype(str).tolist(), normalized_status))
    current_pattern = get_current_user_pattern(user_history, max_suffix_len=max_suffix_len)
    current_event_pattern = get_current_user_event_pattern(user_history, max_suffix_len=max_suffix_len)

    print(f"👤 [User Pattern] Current sequence: {user_events}")
    print(f"👤 [User Pattern] Pattern window: {current_event_pattern}")

    if len(user_events) < 2:
        fallback = user_events[-1][0] if user_events else None
        return fallback, {
            'pattern': current_pattern,
            'pattern_with_status': current_event_pattern,
            'matched_suffix_len': 0,
            'candidate_counts': [],
            'selected_function': fallback,
        }

    max_suffix_len = min(max_suffix_len, len(user_events) - 1)
    for suffix_len in range(max_suffix_len, 0, -1):
        suffix = tuple(user_events[-suffix_len:])
        next_functions = []

        for idx in range(len(user_events) - suffix_len):
            if tuple(user_events[idx:idx + suffix_len]) == suffix:
                next_functions.append(user_events[idx + suffix_len][0])

        if not next_functions:
            continue

        counts = Counter(next_functions)
        ranked_next_functions = [
            {'function': function_name, 'count': count}
            for function_name, count in counts.most_common()
        ]
        selected_function = ranked_next_functions[0]['function']
        matched_pattern = [function_name for function_name, _ in suffix]

        print(f"✅ [User Pattern] Matched suffix: {list(suffix)}")
        print(f"✅ [User Pattern] Next-function counts: {ranked_next_functions}")
        print(f"✅ [User Pattern] Selected warmup function: {selected_function}")

        return selected_function, {
            'pattern': matched_pattern,
            'pattern_with_status': list(suffix),
            'matched_suffix_len': suffix_len,
            'candidate_counts': ranked_next_functions,
            'selected_function': selected_function,
        }

    fallback = user_events[-1][0]
    print(f"⚠️ [User Pattern] No repeated suffix matched. Falling back to last function: {fallback}")
    return fallback, {
        'pattern': current_pattern,
        'pattern_with_status': current_event_pattern,
        'matched_suffix_len': 0,
        'candidate_counts': [],
        'selected_function': fallback,
    }

# --------------------------------------------------------
# Online Training Logic
# --------------------------------------------------------
def perform_online_learning(data: InvocationData):
    try:
        # read new data from csv
        df_full = pd.read_csv(CSV_PATH)
        user_id = str(data.user)
        
        # get latest rows
        user_history = get_sorted_user_history(df_full, user_id).tail(SEQUENCE_LEN + 1)
        
        if len(user_history) > SEQUENCE_LEN:
            print(f"🔄 [Online Learning] Training model for user: {user_id}")
            
            #create features
            hist = user_history.iloc[:-1]
            f_indices = func_enc.transform(hist['function'].astype(str).values)
            u_idx = user_enc.transform([user_id])[0]
            t_indices = encode_hour_series_to_time_idx(hist['hour'])
            s_indices = pd.to_numeric(hist['status'], errors='coerce').fillna(0).clip(0, 1).astype(np.int32).values
            
            #Target/Label
            actual_next_func = user_history.iloc[-1]['function']
            target_idx = func_enc.transform([actual_next_func])[0]
            
            # Shapes
            X_func = np.array([f_indices], dtype=np.int32)
            X_user = np.array([[u_idx] * SEQUENCE_LEN], dtype=np.int32)
            X_time = np.array([t_indices], dtype=np.int32)
            X_status = np.array([s_indices], dtype=np.int32)
            
            # One-hot encoding for the label
            num_classes = len(func_enc.classes_)
            Y_one_hot = to_categorical([target_idx], num_classes=num_classes)

            # Model Update (Using Tensors for Thread Safety)
            with tf.device('/CPU:0'):
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                model.train_on_batch(
                    [tf.convert_to_tensor(X_func, dtype=tf.int32), 
                     tf.convert_to_tensor(X_user, dtype=tf.int32), 
                     tf.convert_to_tensor(X_time, dtype=tf.int32),
                     tf.convert_to_tensor(X_status, dtype=tf.int32)],
                    tf.convert_to_tensor(Y_one_hot, dtype=tf.float32)
                )
                model.save(MODEL_PATH)
                
            print(f"✅ [Online Learning] Model updated with '{actual_next_func}' pattern.")

    except Exception as e:
        print(f"❌ [Online Learning Error] {e}")


def should_trigger_online_learning(total_user_events: int) -> bool:
    # Trigger only on each 10th invocation, and only when sequence+target exists.
    return total_user_events >= (SEQUENCE_LEN + 1) and total_user_events % ONLINE_LEARN_INTERVAL == 0

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def append_to_csv(data: InvocationData):
    try:
        dt = datetime.fromtimestamp(data.timestamp / 1000.0)
        day = dt.strftime('%#m/%#d/%Y')
        hour = dt.strftime('%#I:%M:%S %p')
        status_value = data.functionStatus if data.functionStatus is not None else data.function_status
        new_row = [data.function, data.user, day, hour, data.timestamp, data.timestamp, int(status_value)]
        with open(CSV_PATH, mode='a', newline='') as f:
            csv.writer(f).writerow(new_row)
    except Exception as e:
        print(f"❌ [CSV Error] {e}")

def trigger_warmup(func_name: str):
    url = f"{GATEWAY_URL}/function/{func_name}"
    try:
        requests.post(url, json={}, timeout=0.5)
        print(f"🔥 [Action] Warm-up Triggered: {func_name}")
    except:
        pass

# --------------------------------------------------------
# Predict Endpoint
# --------------------------------------------------------
@app.post("/predict")
async def predict(data: InvocationData, background_tasks: BackgroundTasks):
    try:
        data.function = normalize_function_name(data.function)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # add data to csv
    append_to_csv(data)
    
    try:
        df = pd.read_csv(CSV_PATH)
        user_id = str(data.user)
        user_df = get_sorted_user_history(df, user_id)
        user_event_count = len(user_df)
        user_history = user_df.tail(SEQUENCE_LEN)
        function_counts = df['function'].astype(str).value_counts()

        online_learning_triggered = False
        if should_trigger_online_learning(user_event_count):
            background_tasks.add_task(perform_online_learning, data)
            online_learning_triggered = True
            print(
                f"🔄 [Online Learning] Scheduled for user {user_id} "
                f"at event count {user_event_count}."
            )
        
        if len(user_history) < SEQUENCE_LEN:
            return {
                "prediction": "None",
                "confidence": 0.0,
                "status": "Need more data",
                "warmup_triggered": False,
                "online_learning_triggered": online_learning_triggered,
            }

        # Data Preprocessing
        f_indices = func_enc.transform(user_history['function'].astype(str).values).reshape(1, SEQUENCE_LEN)
        u_idx = user_enc.transform([user_id])[0]
        u_seq = np.array([[u_idx] * SEQUENCE_LEN], dtype=np.int32)
        t_seq = encode_hour_series_to_time_idx(user_history['hour']).reshape(1, SEQUENCE_LEN)
        s_seq = (
            pd.to_numeric(user_history['status'], errors='coerce')
            .fillna(0)
            .clip(0, 1)
            .astype(np.int32)
            .to_numpy()
            .reshape(1, SEQUENCE_LEN)
        )
        f_indices = f_indices.astype(np.int32)

        # Prediction
        with tf.device('/CPU:0'):
            pred = model.predict([f_indices, u_seq, t_seq, s_seq], verbose=0)
            
        prob_vector = pred[0]
        p_idx = np.argmax(prob_vector)
        confidence = np.max(prob_vector)
        predicted_func = func_enc.inverse_transform([p_idx])[0]
        top2_predictions = get_top_k_predictions(prob_vector, func_enc, k=2)
        warmup_func, pattern_warmup_details = choose_warmup_function_from_pattern(user_df)
        top2_functions = [item["function"] for item in top2_predictions]
        top2_values = [float(item["value"]) for item in top2_predictions]
        warmup_targets, warmup_reason = get_warmup_targets(top2_functions, warmup_func, function_counts)
        print(f"🤖 [Model Top-2] {top2_predictions}")
        print(f"🔥 [Warm-up Plan] reason={warmup_reason} targets={warmup_targets}")

        # Trigger Warm-up
        for target_function in warmup_targets:
            background_tasks.add_task(trigger_warmup, target_function)

        return {
            "prediction": predicted_func, 
            "confidence": float(confidence),
            "value": float(confidence),
            "top2_predictions": top2_predictions,
            "top2_functions": top2_functions,
            "top2_values": top2_values,
            "warmup_function": warmup_func,
            "warmup_functions": warmup_targets,
            "warmup_reason": warmup_reason,
            "warmup_candidates": pattern_warmup_details['candidate_counts'],
            "current_pattern": pattern_warmup_details['pattern'],
            "matched_suffix_len": pattern_warmup_details['matched_suffix_len'],
            "warmup_triggered": bool(warmup_targets),
            "online_learning_triggered": online_learning_triggered,
            "status": "Success"
        }

    except Exception as e:
        print(f"❌ [Prediction Error] {e}")
        return {"error": str(e), "warmup_triggered": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
