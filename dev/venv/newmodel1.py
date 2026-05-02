import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout, Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical

# Custom Attention Layer
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

# Data Preparation and model building function
def prepare_and_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '..', 'gru_dataset_fixed_dates_8000.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    print(f"[Info] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df = df.sort_values(['user', 't0']).reset_index(drop=True)

    # Encoders
    f_enc, u_enc, t_enc = LabelEncoder(), LabelEncoder(), LabelEncoder()
    
    df['func_idx'] = f_enc.fit_transform(df['function'])
    df['user_idx'] = u_enc.fit_transform(df['user'].astype(str))
    
    # Time formatting
    df['hour_dt'] = pd.to_datetime(df['hour'].astype(str).str.replace('.', ':', regex=False), format='%I:%M:%S %p', errors='coerce')
    if df['hour_dt'].isna().any():
        bad_values = df.loc[df['hour_dt'].isna(), 'hour'].astype(str).head(5).tolist()
        raise ValueError(f"Unparseable 'hour' values found, examples: {bad_values}")
    df['time_idx'] = t_enc.fit_transform(df['hour_dt'].dt.strftime('%H:%M:%S'))
    
    # Status (convert to integer)
    df['status'] = df['status'].astype(int)

    num_classes = len(f_enc.classes_)
    seq_len = 10

    # create Sequence 
    X_f, X_u, X_t, X_s, y = [], [], [], [], []
    for i in range(len(df) - seq_len):
        X_f.append(df['func_idx'].iloc[i:i+seq_len].values)
        X_u.append(df['user_idx'].iloc[i:i+seq_len].values)
        X_t.append(df['time_idx'].iloc[i:i+seq_len].values)
        X_s.append(df['status'].iloc[i:i+seq_len].values)
        y.append(df['func_idx'].iloc[i+seq_len])

    if not y:
        raise ValueError("No training sequences were created. Check dataset size and sequence length.")

    X = [
        np.array(X_f, dtype=np.int32),
        np.array(X_u, dtype=np.int32),
        np.array(X_t, dtype=np.int32),
        np.array(X_s, dtype=np.int32),
    ]
    y = to_categorical(y, num_classes=num_classes)

    # Model Architecture (4 Inputs) 
    in_f = Input(shape=(seq_len,), name="Func_In")
    in_u = Input(shape=(seq_len,), name="User_In")
    in_t = Input(shape=(seq_len,), name="Time_In")
    in_s = Input(shape=(seq_len,), name="Status_In")

    emb_f = Embedding(num_classes, 64)(in_f)
    emb_u = Embedding(len(u_enc.classes_), 32)(in_u)
    emb_t = Embedding(len(t_enc.classes_), 32)(in_t)
    emb_s = Embedding(2, 8)(in_s) # 0 and 1 status

    merged = Concatenate()([emb_f, emb_u, emb_t, emb_s])
    
    gru_out = GRU(128, return_sequences=True)(merged)
    att_out = SimpleAttention()(gru_out)
    
    x = Dense(64, activation='relu')(att_out)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[in_f, in_u, in_t, in_s], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    print(f"[Info] Starting training with {len(y)} sequences.")
    model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2)
    
    # Save Model and Encoders
    model_path = os.path.join(base_dir, 'gru_status_model2.h5')
    enc_path = os.path.join(base_dir, 'encoders2.pkl')
    model.save(model_path)
    with open(enc_path, 'wb') as f:
        pickle.dump({'f': f_enc, 'u': u_enc, 't': t_enc}, f)

    print(f"[Info] Model saved to: {model_path}")
    print(f"[Info] Encoders saved to: {enc_path}")
    
    return model, f_enc

if __name__ == "__main__":
    model, f_enc = prepare_and_train()