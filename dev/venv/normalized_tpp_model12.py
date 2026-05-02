import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow.keras.backend as K
import pickle
import os

# --- 1. දත්ත පූරණය සහ TPP Preprocessing ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'gru_dataset_fixed_dates_8000.csv')
df = pd.read_csv(csv_path)

# කාලය අනුව පිළිවෙල කිරීම
df = df.sort_values(['user', 't0']).reset_index(drop=True)

# Inter-arrival time (tau) ගණනය කිරීම
df['tau'] = df.groupby('user')['t0'].diff().fillna(0)

# Label Encoding
func_enc = LabelEncoder()
df['func_idx'] = func_enc.fit_transform(df['function'])
num_classes = len(func_enc.classes_)

# Tau Scaling (TPP වලදී කුඩා අගයන් තිබීම පුහුණුවට පහසුයි)
tau_scaler = MinMaxScaler()
df['tau_scaled'] = tau_scaler.fit_transform(df[['tau']])

# --- 2. Sequence Windowing logic ---
def create_tpp_sequences(df, seq_length=10):
    X_func, X_tau, y_mark, y_tau = [], [], [], []
    for i in range(len(df) - seq_length):
        X_func.append(df['func_idx'].iloc[i:i+seq_length].values)
        X_tau.append(df['tau_scaled'].iloc[i:i+seq_length].values)
        y_mark.append(to_categorical(df['func_idx'].iloc[i+seq_length], num_classes))
        y_tau.append(df['tau_scaled'].iloc[i+seq_length])
    return np.array(X_func), np.expand_dims(np.array(X_tau), -1), np.array(y_mark), np.array(y_tau)

X_f, X_t, y_m, y_t = create_tpp_sequences(df)

# --- 3. Custom TPP Loss (Negative Log-Likelihood) ---
def tpp_nll_loss(y_true, y_pred):
    # Simplified NLL based on point process intensity
    rate = y_pred
    tau = y_true
    return -K.mean(K.log(rate + 1e-10) - rate * tau)

# --- 4. Neural TPP Model Architecture ---
def build_neural_tpp(n_func, seq_len):
    input_f = Input(shape=(seq_len,), name="Func_History")
    input_tau = Input(shape=(seq_len, 1), name="Tau_History")

    emb_f = Embedding(n_func, 32)(input_f)
    merged = Concatenate()([emb_f, input_tau])

    gru_out = GRU(64, return_sequences=False)(merged)
    gru_out = Dropout(0.2)(gru_out)
    
    # Mark Prediction (Which function?)
    mark_out = Dense(n_func, activation='softmax', name="mark_output")(gru_out)
    
    # Time Prediction (Intensity rate lambda)
    # Softplus භාවිතා කරන්නේ අගය සැමවිටම ධන (+) අගයක් විය යුතු නිසා
    rate_out = Dense(1, activation='softplus', name="time_output")(gru_out)

    model = Model(inputs=[input_f, input_tau], outputs=[mark_out, rate_out])
    return model

# --- 5. Training with TimeSeriesSplit ---
tscv = TimeSeriesSplit(n_splits=5)
fold = 1

for train_idx, val_idx in tscv.split(X_f):
    print(f"\n--- Training Fold {fold} ---")
    model = build_neural_tpp(num_classes, 10)
    model.compile(
        optimizer='adam',
        loss={"mark_output": "categorical_crossentropy", "time_output": tpp_nll_loss},
        loss_weights={"mark_output": 1.0, "time_output": 0.5}
    )
    
    model.fit(
        [X_f[train_idx], X_t[train_idx]], 
        {"mark_output": y_m[train_idx], "time_output": y_t[train_idx]},
        validation_data=([X_f[val_idx], X_t[val_idx]], {"mark_output": y_m[val_idx], "time_output": y_t[val_idx]}),
        epochs=20, batch_size=64, verbose=1
    )
    fold += 1

# --- 6. Saving ---
model.save('neural_tpp_model.h5')
with open('tau_scaler.pkl', 'wb') as f:
    pickle.dump(tau_scaler, f)

print("\nFull Neural TPP Model training complete!")