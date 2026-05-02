import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout, Layer
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# --- 1. Custom Attention Layer ---
# This layer allows the model to focus on specific important functions in the sequence,
# which is key for identifying infrequent but predictive patterns.
class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        # Alignment scores
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Weighting
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config()
        return config

# --- 2. Data Loading & Preprocessing ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'gru_dataset_fixed_dates_8000.csv')

df = pd.read_csv(csv_path)
df = df.sort_values(['user', 't0']).reset_index(drop=True)

func_enc = LabelEncoder()
user_enc = LabelEncoder()
time_enc = LabelEncoder()

df['func_idx'] = func_enc.fit_transform(df['function'])
df['user_idx'] = user_enc.fit_transform(df['user'].astype(str))

# Time Parsing
hour_clean = df['hour'].astype(str).str.strip().str.replace('.', ':', regex=False).str.replace(r'\s+', ' ', regex=True)
df['hour_dt'] = pd.to_datetime(hour_clean, format='%I:%M:%S %p', errors='coerce')

if df['hour_dt'].isna().any():
    bad_hours = df.loc[df['hour_dt'].isna(), 'hour'].astype(str).head(5).tolist()
    raise ValueError(f"Unparseable time values: {bad_hours}")

df['time_idx'] = time_enc.fit_transform(df['hour_dt'].dt.strftime('%H:%M:%S'))

num_classes = len(func_enc.classes_)
y_all = to_categorical(df['func_idx'], num_classes=num_classes)

# --- 3. Windowing (Sequence Creation) ---

def create_sequences(df, y_data, seq_length=10):
    X_f, X_u, X_t, y = [], [], [], []
    for i in range(len(df) - seq_length):
        X_f.append(df['func_idx'].iloc[i:i+seq_length].values)
        X_u.append(df['user_idx'].iloc[i:i+seq_length].values)
        X_t.append(df['time_idx'].iloc[i:i+seq_length].values)
        y.append(y_data[i+seq_length])
    return np.array(X_f), np.array(X_u), np.array(X_t), np.array(y)

sequence_length = 10
X_func, X_user, X_time, y_func = create_sequences(df, y_all, sequence_length)

# --- 4. Model Architecture (GRU + Attention) ---

def build_attention_gru_model(n_func, n_user, n_time, seq_len):
    input_f = Input(shape=(seq_len,), name="Func_Input")
    input_u = Input(shape=(seq_len,), name="User_Input")
    input_t = Input(shape=(seq_len,), name="Time_Input")

    # Embeddings
    emb_f = Embedding(n_func, 32)(input_f) 
    emb_u = Embedding(n_user, 32)(input_u)
    emb_t = Embedding(n_time, 32)(input_t)

    merged = Concatenate()([emb_f, emb_u, emb_t])

    # GRU with return_sequences=True to feed the Attention layer
    gru_out = GRU(64, return_sequences=True)(merged)
    gru_out = Dropout(0.5)(gru_out)
    
    # Custom Attention Layer
    attention_out = SimpleAttention()(gru_out)
    
    # Final classifier
    x = Dense(32, activation='relu')(attention_out)
    output = Dense(n_func, activation='softmax')(x)

    model = Model(inputs=[input_f, input_u, input_t], outputs=output)
    return model

# --- 5. Training ---

# Calculate Class Weights for Focal Loss
y_integers = np.argmax(y_func, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
alpha_weights = weights.tolist()

split_idx = int(len(X_func) * 0.8)
X_f_tr, X_f_val = X_func[:split_idx], X_func[split_idx:]
X_u_tr, X_u_val = X_user[:split_idx], X_user[split_idx:]
X_t_tr, X_t_val = X_time[:split_idx], X_time[split_idx:]
y_tr, y_val = y_func[:split_idx], y_func[split_idx:]

model = build_attention_gru_model(len(func_enc.classes_), len(user_enc.classes_), len(time_enc.classes_), sequence_length)

focal_loss = CategoricalFocalCrossentropy(alpha=alpha_weights, gamma=2.0)
model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
]

print("\n--- Training Attention-GRU Model ---")
model.fit(
    [X_f_tr, X_u_tr, X_t_tr], y_tr,
    validation_data=([X_f_val, X_u_val, X_t_val], y_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# --- 6. Saving Model & Encoders ---

# Save the model (including the custom Attention layer)
model.save('gru_attention_model1.h5')

# Save Encoders
with open('func_encoder.pkl', 'wb') as f:
    pickle.dump(func_enc, f)
with open('user_encoder.pkl', 'wb') as f:
    pickle.dump(user_enc, f)
with open('time_encoder.pkl', 'wb') as f:
    pickle.dump(time_enc, f)

print("\nModel and encoders saved successfully!")

# --- How to Load later ---
# from tensorflow.keras.models import load_model
# model = load_model('gru_attention_model.h5', custom_objects={'SimpleAttention': SimpleAttention})