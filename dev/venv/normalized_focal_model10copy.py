import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import os

# --- 1. දත්ත පූරණය සහ පූර්ව සැකසුම (Data Loading & Preprocessing) ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'gru_dataset_fixed_dates_8000.csv')

df = pd.read_csv(csv_path)

# කාලය අනුව දත්ත පිළිවෙල කිරීම
df = df.sort_values(['user', 't0']).reset_index(drop=True)

# Label Encoding
func_enc = LabelEncoder()
user_enc = LabelEncoder()
time_enc = LabelEncoder()

df['func_idx'] = func_enc.fit_transform(df['function'])
df['user_idx'] = user_enc.fit_transform(df['user'].astype(str))

hour_clean = (
    df['hour']
    .astype(str)
    .str.strip()
    .str.replace('.', ':', regex=False)
    .str.replace(r'\s+', ' ', regex=True)
)

df['hour_dt'] = pd.to_datetime(hour_clean, format='%I:%M:%S %p', errors='coerce')
if df['hour_dt'].isna().any():
    bad_hours = df.loc[df['hour_dt'].isna(), 'hour'].astype(str).head(10).tolist()
    raise ValueError(f"Unparseable 'hour' values found (showing up to 10): {bad_hours}")

df['time_idx'] = time_enc.fit_transform(df['hour_dt'].dt.strftime('%H:%M:%S'))

# Target Variable (y) සකස් කිරීම
num_classes = len(func_enc.classes_)
y_all = to_categorical(df['func_idx'], num_classes=num_classes)

# --- 2. Sequence එකක් ලෙස දත්ත සැකසීම (Windowing) ---

def create_sequences(df, y_data, seq_length=10):
    X_f, X_u, X_t, y = [], [], [], []
    for i in range(len(df) - seq_length):
        X_f.append(df['func_idx'].iloc[i:i+seq_length].values)
        X_u.append(df['user_idx'].iloc[i:i+seq_length].values)
        X_t.append(df['time_idx'].iloc[i:i+seq_length].values)
        y.append(y_data[i+seq_length]) # ඊළඟ පියවර පුරෝකථනය කිරීමට
    return np.array(X_f), np.array(X_u), np.array(X_t), np.array(y)

sequence_length = 10
X_func, X_user, X_time, y_func = create_sequences(df, y_all, sequence_length)

# --- 3. Model Architecture එක නිර්මාණය (build_sequence_model) ---

def build_sequence_model(n_func, n_user, n_time, seq_len):
    # Inputs
    input_f = Input(shape=(seq_len,), name="Func_Input")
    input_u = Input(shape=(seq_len,), name="User_Input")
    input_t = Input(shape=(seq_len,), name="Time_Input")

    # Embeddings (දත්ත වල ඇති සම්බන්ධතා හඳුනා ගැනීමට)
    emb_f = Embedding(n_func, 32)(input_f)
    emb_u = Embedding(n_user, 16)(input_u)
    emb_t = Embedding(n_time, 16)(input_t)

    # Concatenate features
    merged = Concatenate()([emb_f, emb_u, emb_t])

    # GRU Layer
    gru_out = GRU(64, return_sequences=False)(merged)
    gru_out = Dropout(0.2)(gru_out)
    
    # Final Dense Layer
    output = Dense(n_func, activation='softmax')(gru_out)

    model = Model(inputs=[input_f, input_u, input_t], outputs=output)
    return model

# --- 4. Cross-Validation සහ Training ---

# Class Weights ගණනය කිරීම
y_integers = np.argmax(y_func, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
alpha_weights = weights.tolist()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

split_idx = int(len(X_func) * 0.8)
if split_idx <= 0 or split_idx >= len(X_func):
    raise ValueError("Not enough sequence data to create train/validation split.")

X_f_tr, X_f_val = X_func[:split_idx], X_func[split_idx:]
X_u_tr, X_u_val = X_user[:split_idx], X_user[split_idx:]
X_t_tr, X_t_val = X_time[:split_idx], X_time[split_idx:]
y_tr, y_val = y_func[:split_idx], y_func[split_idx:]

print("\n--- Training (single chronological split) ---")

model = build_sequence_model(len(func_enc.classes_), len(user_enc.classes_), len(time_enc.classes_), sequence_length)
focal_loss = CategoricalFocalCrossentropy(alpha=alpha_weights, gamma=2.0)

model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])

model.fit(
    [X_f_tr, X_u_tr, X_t_tr], y_tr,
    validation_data=([X_f_val, X_u_val, X_t_val], y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- 5. Saving Model & Encoders ---

model.save('gru_focal_loss_model10copy.h5')

with open('func_encoder10copy.pkl', 'wb') as f:
    pickle.dump(func_enc, f)
with open('user_encoder10copy.pkl', 'wb') as f:
    pickle.dump(user_enc, f)
with open('time_encoder10copy.pkl', 'wb') as f:
    pickle.dump(time_enc, f)

print("\nModel and encoders saved successfully!")