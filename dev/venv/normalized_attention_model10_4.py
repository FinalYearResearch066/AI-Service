import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout, Layer
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import os

# --- 1. Custom Attention Layer ---
# මෙය sequence එකේ ඇති වැදගත්ම පියවරවල් හඳුනා ගැනීමට උදවු වේ.
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
        # Score එකක් ගණනය කර Softmax මගින් weight එකක් ලබා දෙයි
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# --- 2. Data Loading & Preprocessing ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'gru_dataset_fixed_dates_8000.csv')
df = pd.read_csv(csv_path)

# කාලය අනුව දත්ත පිළිවෙල කිරීම
df = df.sort_values(['user', 't0']).reset_index(drop=True)

# Encoding
func_enc = LabelEncoder()
user_enc = LabelEncoder()
time_enc = LabelEncoder()

df['func_idx'] = func_enc.fit_transform(df['function'])
df['user_idx'] = user_enc.fit_transform(df['user'].astype(str))

# Time feature processing
hour_clean = df['hour'].astype(str).str.strip().str.replace('.', ':', regex=False)
df['hour_dt'] = pd.to_datetime(hour_clean, format='%I:%M:%S %p', errors='coerce')
df['time_idx'] = time_enc.fit_transform(df['hour_dt'].dt.strftime('%H:%M:%S'))

num_classes = len(func_enc.classes_)
y_all = to_categorical(df['func_idx'], num_classes=num_classes)

# Sequence windowing (පියවර 10 ක මතකයක්)
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

# --- 3. Attention-based GRU Model Architecture ---
def build_attention_model(n_func, n_user, n_time, seq_len):
    in_f = Input(shape=(seq_len,), name="Func_Input")
    in_u = Input(shape=(seq_len,), name="User_Input")
    in_t = Input(shape=(seq_len,), name="Time_Input")

    # Embeddings
    emb_f = Embedding(n_func, 32)(in_f)
    emb_u = Embedding(n_user, 16)(in_u)
    emb_t = Embedding(n_time, 16)(in_t)

    merged = Concatenate()([emb_f, emb_u, emb_t])

    # GRU with return_sequences=True to feed Attention Layer
    gru_out = GRU(64, return_sequences=True)(merged)
    
    # Attention Layer
    att_out = SimpleAttention()(gru_out)
    
    x = Dropout(0.2)(att_out)
    output = Dense(n_func, activation='softmax')(x)

    model = Model(inputs=[in_f, in_u, in_t], outputs=output)
    return model

# --- 4. Training with Focal Loss & TimeSplit ---
# Class weights ගණනය කිරීම
y_integers = np.argmax(y_func, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
alpha_weights = weights.tolist()

tscv = TimeSeriesSplit(n_splits=5)
fold = 1

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

for train_idx, val_idx in tscv.split(X_func):
    print(f"\n--- Training Fold {fold} (Attention Mode) ---")
    
    X_tr = [X_func[train_idx], X_user[train_idx], X_time[train_idx]]
    X_val = [X_func[val_idx], X_user[val_idx], X_time[val_idx]]
    y_tr, y_val = y_func[train_idx], y_func[val_idx]

    model = build_attention_model(len(func_enc.classes_), len(user_enc.classes_), len(time_enc.classes_), sequence_length)

    # Gamma=5.0 යොදා ඇත්තේ අමාරු/දුර්ලභ දත්ත වලට වැඩි අවධානයක් දීමටයි
    focal_loss = CategoricalFocalCrossentropy(alpha=alpha_weights, gamma=5.0)
    model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    fold += 1

# --- 5. Saving Results ---
model.save('gru_attention_focal_model10_4.h5')
with open('func_encoder10_4.pkl', 'wb') as f:
    pickle.dump(func_enc, f)
with open('user_encoder10_4.pkl', 'wb') as f:
    pickle.dump(user_enc, f)
with open('time_encoder10_4.pkl', 'wb') as f:
    pickle.dump(time_enc, f)


print("\nAttention-based Model and Encoders saved successfully!")