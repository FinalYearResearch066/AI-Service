import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# --- 1. දත්ත පූරණය සහ Croston Preprocessing ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'gru_dataset_fixed_dates_8000.csv')
df = pd.read_csv(csv_path)

# කාලය අනුව පිළිවෙල කිරීම (Sorting)
df = df.sort_values(['user', 't0']).reset_index(drop=True)

# කාල පරතරය (Interval) ගණනය කිරීම - Croston's Logic
# එක් invocation එකක සිට ඊළඟ එකට ඇති කාලය (තත්පර වලින් හෝ t0 අගයෙන්)
df['interval'] = df.groupby('user')['t0'].diff().fillna(0)

# Label Encoding
func_enc = LabelEncoder()
user_enc = LabelEncoder()
time_enc = LabelEncoder()

df['func_idx'] = func_enc.fit_transform(df['function'])
df['user_idx'] = user_enc.fit_transform(df['user'].astype(str))

# කාලය පූර්ව සැකසුම (Hour cleaning)
hour_clean = df['hour'].astype(str).str.strip().str.replace('.', ':', regex=False)
df['hour_dt'] = pd.to_datetime(hour_clean, format='%I:%M:%S %p', errors='coerce')
if df['hour_dt'].isna().any():
    bad_hours = df.loc[df['hour_dt'].isna(), 'hour'].astype(str).head(10).tolist()
    raise ValueError(f"Unparseable 'hour' values found (showing up to 10): {bad_hours}")
df['time_idx'] = time_enc.fit_transform(df['hour_dt'].dt.strftime('%H:%M:%S'))

# Interval Scaling (Regression සඳහා අත්‍යවශ්‍යයි)
interval_scaler = MinMaxScaler()
df['interval_scaled'] = interval_scaler.fit_transform(df[['interval']])

# Target Variables
num_classes = len(func_enc.classes_)
y_class_all = to_categorical(df['func_idx'], num_classes=num_classes)
y_interval_all = df['interval_scaled'].values

# --- 2. Sequence Windowing (Multi-Output සඳහා) ---

def create_sequences_croston(df, y_class, y_interval, seq_length=10):
    X_f, X_u, X_t, y_c, y_i = [], [], [], [], []
    for i in range(len(df) - seq_length):
        X_f.append(df['func_idx'].iloc[i:i+seq_length].values)
        X_u.append(df['user_idx'].iloc[i:i+seq_length].values)
        X_t.append(df['time_idx'].iloc[i:i+seq_length].values)
        
        # Targets: Next Function Class AND Next Interval
        y_c.append(y_class[i+seq_length])
        y_i.append(y_interval[i+seq_length])
        
    return np.array(X_f), np.array(X_u), np.array(X_t), np.array(y_c), np.array(y_i)

sequence_length = 10
X_func, X_user, X_time, y_class, y_interval = create_sequences_croston(df, y_class_all, y_interval_all, sequence_length)

# --- 3. Neural Renewal Model (Croston-inspired Architecture) ---

def build_neural_croston_model(n_func, n_user, n_time, seq_len):
    input_f = Input(shape=(seq_len,), name="Func_Input")
    input_u = Input(shape=(seq_len,), name="User_Input")
    input_t = Input(shape=(seq_len,), name="Time_Input")

    emb_f = Embedding(n_func, 32)(input_f)
    emb_u = Embedding(n_user, 16)(input_u)
    emb_t = Embedding(n_time, 16)(input_t)

    merged = Concatenate()([emb_f, emb_u, emb_t])
    gru_out = GRU(64, return_sequences=False)(merged)
    gru_out = Dropout(0.2)(gru_out)
    
    # Head 1: Classification (Which function?) - Standard Crossentropy
    class_output = Dense(n_func, activation='softmax', name="class_output")(gru_out)
    
    # Head 2: Regression (When? / Interval) - Mean Squared Error
    interval_output = Dense(1, activation='linear', name="interval_output")(gru_out)

    model = Model(inputs=[input_f, input_u, input_t], outputs=[class_output, interval_output])
    return model

# --- 4. Cross-Validation සහ Training ---

tscv = TimeSeriesSplit(n_splits=5)
fold = 1

for train_idx, val_idx in tscv.split(X_func):
    print(f"\n--- Training Fold {fold} (Neural Croston Mode) ---")
    
    # Data Split
    X_tr = [X_func[train_idx], X_user[train_idx], X_time[train_idx]]
    y_tr = {"class_output": y_class[train_idx], "interval_output": y_interval[train_idx]}
    
    X_val = [X_func[val_idx], X_user[val_idx], X_time[val_idx]]
    y_val = {"class_output": y_class[val_idx], "interval_output": y_interval[val_idx]}

    model = build_neural_croston_model(len(func_enc.classes_), len(user_enc.classes_), len(time_enc.classes_), sequence_length)

    # Multi-loss Compilation
    model.compile(
        optimizer='adam',
        loss={
            "class_output": "categorical_crossentropy", 
            "interval_output": "mse" # කාලය සඳහා Mean Squared Error
        },
        loss_weights={
            "class_output": 1.0, 
            "interval_output": 0.5 # අවශ්‍යතාවය අනුව වෙනස් කළ හැක
        },
        metrics={"class_output": "accuracy", "interval_output": "mae"}
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        verbose=1
    )
    fold += 1

# --- 5. Saving Model & Meta ---

model.save('gru_neural_croston_model11.h5')
with open('interval_scaler.pkl', 'wb') as f:
    pickle.dump(interval_scaler, f)

print("\nNeural Croston model and scalers saved successfully!")