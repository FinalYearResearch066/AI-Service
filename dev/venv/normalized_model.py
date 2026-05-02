import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
import tensorflow as tf
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'dataset1.csv')

df = pd.read_csv(csv_path)

# =========================
# 1️⃣ Load dataset
# =========================
print("Script location:", BASE_DIR)
print("CSV path:", csv_path)

df = pd.read_csv(csv_path)

# =========================
# 2️⃣ Encode categorical features
# =========================
func_enc = LabelEncoder()
user_enc = LabelEncoder()
df['func_idx'] = func_enc.fit_transform(df['function'])
df['user_idx'] = user_enc.fit_transform(df['user'])

# =========================
# 3️⃣ Extract time features
# =========================
df['day'] = pd.to_datetime(df['day'], format='%m/%d/%Y')
df['day_of_week'] = df['day'].dt.weekday  # 0 = Monday, 6 = Sunday

# Convert hour column to datetime to extract hour and minute
df['hour_dt'] = pd.to_datetime(df['hour'], format='%I:%M:%S %p')
df['hour'] = df['hour_dt'].dt.hour
df['minute'] = df['hour_dt'].dt.minute

# =========================
# 4️⃣ Compute function duration 
# =========================
# Assuming t0/t5 are in Unix timestamp (ms)
df['duration'] = (df['t5'] - df['t0']) / 1000.0  # convert ms -> seconds

# =========================
# 5️⃣ Normalize numeric features
# =========================
scaler = StandardScaler()
df[['hour_s', 'minute_s', 'day_of_week_s', 'duration_s']] = scaler.fit_transform(
    df[['hour', 'minute', 'day_of_week', 'duration']]
)

# =========================
# 6️⃣ Prepare sequences for GRU
# =========================
sequence_length = 3  # number of previous steps to use
X_func, X_user, X_time, y_func = [], [], [], []

for user_id, user_df in df.groupby('user_idx'):
    user_df = user_df.sort_values('t0')  # sort by timestamp
    func_seq = user_df['func_idx'].values
    time_seq = user_df[['hour_s', 'minute_s', 'day_of_week_s', 'duration_s']].values

    for i in range(len(user_df) - sequence_length):
        X_func.append(func_seq[i:i+sequence_length])
        X_user.append([user_id]*sequence_length)
        X_time.append(time_seq[i:i+sequence_length])
        y_func.append(func_seq[i+sequence_length])  # next function

# Convert to numpy arrays
X_func = np.array(X_func)
X_user = np.array(X_user)
X_time = np.array(X_time)
y_func = to_categorical(y_func, num_classes=len(func_enc.classes_))

print("Sequences prepared:", X_func.shape, X_time.shape, y_func.shape)

# =========================
# 7️⃣ Build GRU model for next function prediction
# =========================
def build_sequence_model(num_funcs, num_users, seq_len, time_features=4):
    # Inputs
    in_func = layers.Input(shape=(seq_len,), name='func_input')
    in_user = layers.Input(shape=(seq_len,), name='user_input')
    in_time = layers.Input(shape=(seq_len, time_features), name='time_input')

    # Embeddings
    emb_f = layers.Embedding(num_funcs, 16)(in_func)
    emb_u = layers.Embedding(num_users, 16)(in_user)

    # Concatenate embeddings with time features
    x = layers.Concatenate()([emb_f, emb_u, in_time])

    # GRU layers
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.GRU(64, dropout=0.2))(x)

    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output: next function prediction
    out = layers.Dense(num_funcs, activation='softmax')(x)

    model = Model(inputs=[in_func, in_user, in_time], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_sequence_model(len(func_enc.classes_), len(user_enc.classes_), sequence_length)

# =========================
# 8️⃣ Train model
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    [X_func, X_user, X_time],
    y_func,
    batch_size=20,
    epochs=10,
    validation_split=0.1,
    callbacks=[early_stop]
)

# =========================
# 9️⃣ Save model
# =========================
model.save('gru_next_function_model.h5')

with open('func_encoder.pkl', 'wb') as f:
    pickle.dump(func_enc, f)
with open('user_encoder.pkl', 'wb') as f:
    pickle.dump(user_enc, f)
with open('time_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ GRU model saved.")
print("✅ Encoders and scaler saved.")
