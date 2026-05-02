import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout, Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- 1. Custom Focal Loss Function ---
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Imbalanced classes කළමනාකරණය කිරීමට භාවිතා කරන Focal Loss ශ්‍රිතය.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # අගයන් 0 සහ 1 අතර රඳවා ගැනීමට clip කිරීම
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Cross Entropy ගණනය කිරීම
        cross_entropy = -y_true * K.log(y_pred)
        
        # Focal Loss හි ප්‍රධාන බර තැබීමේ සාධකය (Weighting factor)
        # පහසුවෙන් අනුමාන කළ හැකි දත්ත වල බලපෑම අඩු කරයි
        weight = y_true * K.pow((1.0 - y_pred), gamma)
        
        loss = alpha * weight * cross_entropy
        return K.sum(loss, axis=1)
    
    return focal_loss_fixed

# --- 2. Custom Attention Layer ---
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

# --- 3. Data Preparation and Model Training ---
def prepare_and_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Dataset එක ඇති ස්ථානය (ඔබේ file path එකට අනුව වෙනස් කරගන්න)
    dataset_path = os.path.join(base_dir, '..', 'gru_dataset_fixed_dates_8000.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    
    # Encoders සකස් කිරීම
    f_enc = LabelEncoder()
    u_enc = LabelEncoder()
    t_enc = LabelEncoder()
    
    df['f_idx'] = f_enc.fit_transform(df['function'])
    df['u_idx'] = u_enc.fit_transform(df['user'].astype(str))
    df['t_idx'] = t_enc.fit_transform(df['hour'].astype(str))
    df['s_idx'] = pd.to_numeric(df['status'], errors='coerce').fillna(0).clip(0, 1).astype(np.int32)
    
    num_classes = len(f_enc.classes_)
    u_classes = len(u_enc.classes_)
    t_classes = len(t_enc.classes_)
    
    seq_len = 10
    X_f, X_u, X_t, X_s, y = [], [], [], [], []

    # Sequence සාදා ගැනීම
    for i in range(seq_len, len(df)):
        X_f.append(df['f_idx'].values[i-seq_len:i])
        X_u.append(df['u_idx'].values[i-seq_len:i])
        X_t.append(df['t_idx'].values[i-seq_len:i])
        X_s.append(df['s_idx'].values[i-seq_len:i])
        y.append(df['f_idx'].values[i])

    X = [np.array(X_f), np.array(X_u), np.array(X_t), np.array(X_s)]
    y = to_categorical(np.array(y), num_classes=num_classes)

    # --- Model Architecture ---
    in_f = Input(shape=(seq_len,), name="Func_In")
    in_u = Input(shape=(seq_len,), name="User_In")
    in_t = Input(shape=(seq_len,), name="Time_In")
    in_s = Input(shape=(seq_len,), name="Status_In")

    emb_f = Embedding(num_classes, 64)(in_f)
    emb_u = Embedding(u_classes, 32)(in_u)
    emb_t = Embedding(t_classes, 32)(in_t)
    emb_s = Embedding(2, 8)(in_s)

    merged = Concatenate()([emb_f, emb_u, emb_t, emb_s])
    
    gru_out = GRU(128, return_sequences=True)(merged)
    att_out = SimpleAttention()(gru_out)
    
    x = Dense(64, activation='relu')(att_out)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[in_f, in_u, in_t, in_s], outputs=output)

    # --- Focal Loss සමඟ Compile කිරීම ---
    model.compile(
        optimizer='adam', 
        loss=focal_loss(gamma=2.0, alpha=0.25), 
        metrics=['accuracy']
    )

    print("[Info] Starting training with Focal Loss...")
    model.fit(X, y, epochs=30, batch_size=64, validation_split=0.2)

    # Model එක සහ Encoders save කිරීම
    model.save(os.path.join(base_dir, 'gru_status_model2.h5'))
    with open(os.path.join(base_dir, 'encoders2.pkl'), 'wb') as f:
        pickle.dump({'f_enc': f_enc, 'u_enc': u_enc, 't_enc': t_enc}, f)
    
    print("[Success] Model saved as gru_status_model2.h5")

if __name__ == "__main__":
    prepare_and_train()