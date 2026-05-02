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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import os

# --- 1. Custom Attention Layer ---
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

# --- 2. Data Preprocessing (කලින් කේතයම වේ) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'gru_dataset_fixed_dates_8000.csv')
df = pd.read_csv(csv_path)
df = df.sort_values(['user', 't0']).reset_index(drop=True)

func_enc, user_enc, time_enc = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['func_idx'] = func_enc.fit_transform(df['function'])
df['user_idx'] = user_enc.fit_transform(df['user'].astype(str))
hour_clean = df['hour'].astype(str).str.strip().str.replace('.', ':', regex=False)
df['hour_dt'] = pd.to_datetime(hour_clean, format='%I:%M:%S %p', errors='coerce')
df['time_idx'] = time_enc.fit_transform(df['hour_dt'].dt.strftime('%H:%M:%S'))

num_classes = len(func_enc.classes_)
y_all = to_categorical(df['func_idx'], num_classes=num_classes)

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

# --- 3. Attention-based Architecture ---
def build_precision_model(n_func, n_user, n_time, seq_len):
    input_f = Input(shape=(seq_len,), name="Func_Input")
    input_u = Input(shape=(seq_len,), name="User_Input")
    input_t = Input(shape=(seq_len,), name="Time_Input")

    emb_f = Embedding(n_func, 32)(input_f)
    emb_u = Embedding(n_user, 16)(input_u)
    emb_t = Embedding(n_time, 16)(input_t)

    merged = Concatenate()([emb_f, emb_u, emb_t])

    # return_sequences=True must be set for Attention
    gru_out = GRU(64, return_sequences=True)(merged)
    att_out = SimpleAttention()(gru_out)
    
    x = Dropout(0.3)(att_out) # Dropout මඳක් වැඩි කරන ලදී Overfitting වැළැක්වීමට
    output = Dense(n_func, activation='softmax')(x)

    model = Model(inputs=[input_f, input_u, input_t], outputs=output)
    return model

# --- 4. Training with Gamma 5.0 ---
y_integers = np.argmax(y_func, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)

# Precision වැඩි කිරීමට දුර්ලභ class වලට තවත් බර වැඩි කිරීම (Manual boost)
# signup සහ forgot-password වල index සොයාගෙන ඒවායේ weights තවත් 20% කින් වැඩි කරන්න
signup_idx = func_enc.transform(['signup'])[0]
forgot_idx = func_enc.transform(['forgot-password'])[0]
weights[signup_idx] *= 1.5
weights[forgot_idx] *= 1.2
alpha_weights = weights.tolist()

model = build_precision_model(num_classes, len(user_enc.classes_), len(time_enc.classes_), sequence_length)
focal_loss = CategoricalFocalCrossentropy(alpha=alpha_weights, gamma=5.0)

model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])

# Train/Val split
split = int(len(X_func) * 0.8)
model.fit(
    [X_func[:split], X_user[:split], X_time[:split]], y_func[:split],
    validation_data=([X_func[split:], X_user[split:], X_time[split:]], y_func[split:]),
    epochs=40, # වැඩි වාර ගණනක් පුහුණු කිරීම
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)],
    verbose=1
)

model.save('gru_attention_precision_optimized.h5')

with open(os.path.join(BASE_DIR, 'func_encoder_optimized.pkl'), 'wb') as f:
    pickle.dump(func_enc, f)

with open(os.path.join(BASE_DIR, 'user_encoder_optimized.pkl'), 'wb') as f:
    pickle.dump(user_enc, f)

with open(os.path.join(BASE_DIR, 'time_encoder_optimized.pkl'), 'wb') as f:
    pickle.dump(time_enc, f)

print("All encoders (Function, User, Time) saved successfully!")
print("\nPrecision optimized model saved!")