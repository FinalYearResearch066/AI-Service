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
from tensorflow.keras.utils import to_categorical

# TensorFlow configuration to avoid "numpy()" error
tf.config.run_functions_eagerly(True)

app = FastAPI()

# ========================================================
# CONFIGURATION
# ========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..","gru_dataset_fixed_dates_8000.csv")
MODEL_PATH = os.path.join(BASE_DIR, "gru_focal_loss_model10copy2.h5")
FUNC_ENC_PATH = os.path.join(BASE_DIR, "func_encoder10copy2.pkl")
USER_ENC_PATH = os.path.join(BASE_DIR,"user_encoder10copy2.pkl")
TIME_ENC_PATH = os.path.join(BASE_DIR,"time_encoder10copy2.pkl")

ONLINE_LEARN_INTERVAL = 10
GATEWAY_URL = "http://127.0.0.1:8080"

# Load ML Assets
model = tf.keras.models.load_model(MODEL_PATH)
with open(FUNC_ENC_PATH, "rb") as f: func_enc = pickle.load(f)
with open(USER_ENC_PATH, "rb") as f: user_enc = pickle.load(f)
with open(TIME_ENC_PATH, "rb") as f: time_enc = pickle.load(f)
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

    # Some clients mistakenly send function input JSON in the `function` field.
    # Try extracting a real function label from common keys.
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

# --------------------------------------------------------
# Online Training Logic
# --------------------------------------------------------
def perform_online_learning(data: InvocationData):
    try:
        # read new data from csv
        df_full = pd.read_csv(CSV_PATH)
        user_id = str(data.user)
        
        # get latest rows
        user_history = df_full[df_full['user'].astype(str) == user_id].tail(SEQUENCE_LEN + 1)
        
        if len(user_history) > SEQUENCE_LEN:
            print(f"🔄 [Online Learning] Training model for user: {user_id}")
            
            #create features
            hist = user_history.iloc[:-1]
            f_indices = func_enc.transform(hist['function'].astype(str).values)
            u_idx = user_enc.transform([user_id])[0]
            t_indices = encode_hour_series_to_time_idx(hist['hour'])
            
            #Target/Label
            actual_next_func = user_history.iloc[-1]['function']
            target_idx = func_enc.transform([actual_next_func])[0]
            
            # Shapes
            X_func = np.array([f_indices], dtype=np.int32)
            X_user = np.array([[u_idx] * SEQUENCE_LEN], dtype=np.int32)
            X_time = np.array([t_indices], dtype=np.int32)
            
            # One-hot encoding for the label
            num_classes = len(func_enc.classes_)
            Y_one_hot = to_categorical([target_idx], num_classes=num_classes)

            # Model Update (Using Tensors for Thread Safety)
            with tf.device('/CPU:0'):
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                model.train_on_batch(
                    [tf.convert_to_tensor(X_func, dtype=tf.int32), 
                     tf.convert_to_tensor(X_user, dtype=tf.int32), 
                     tf.convert_to_tensor(X_time, dtype=tf.int32)],
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
        user_df = df[df['user'].astype(str) == user_id]
        user_event_count = len(user_df)
        user_history = user_df.tail(SEQUENCE_LEN)

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
        f_indices = f_indices.astype(np.int32)

        # Prediction
        with tf.device('/CPU:0'):
            pred = model.predict([f_indices, u_seq, t_seq], verbose=0)
            
        p_idx = np.argmax(pred[0])
        confidence = np.max(pred[0])
        predicted_func = func_enc.inverse_transform([p_idx])[0]

        # Trigger Warm-up
        background_tasks.add_task(trigger_warmup, predicted_func)

        return {
            "prediction": predicted_func, 
            "confidence": float(confidence),
            "warmup_triggered": True,
            "online_learning_triggered": online_learning_triggered,
            "status": "Success"
        }

    except Exception as e:
        print(f"❌ [Prediction Error] {e}")
        return {"error": str(e), "warmup_triggered": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
