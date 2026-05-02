from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
import tensorflow as tf
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'gru_dataset_fixed_dates_8000.csv')

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
df['day_of_week'] = df['day'].dt.weekday

hour_clean = (
    df['hour']
    .astype(str)
    .str.strip()
    .str.replace('.', ':', regex=False)
    .str.replace(r'\s+', ' ', regex=True)
)

df['hour_dt'] = pd.to_datetime(hour_clean, format='%I:%M:%S %p', errors='coerce')

invalid_hour_mask = df['hour_dt'].isna()
if invalid_hour_mask.any():
    bad_hours = df.loc[invalid_hour_mask, 'hour'].astype(str).head(10).tolist()
    raise ValueError(f"Unparseable 'hour' values found after cleanup (showing up to 10): {bad_hours}")

df['hour'] = df['hour_dt'].dt.hour
df['minute'] = df['hour_dt'].dt.minute

# =========================
# 4️⃣ Compute duration
# =========================
df['duration'] = (df['t5'] - df['t0']) / 1000.0

# =========================
# 5️⃣ Normalize numeric features
# =========================
scaler = StandardScaler()

df[['hour_s','minute_s','day_of_week_s','duration_s']] = scaler.fit_transform(
    df[['hour','minute','day_of_week','duration']]
)

# =========================
# 6️⃣ Prepare sequences
# =========================
sequence_length = 5

X_func = []
X_user = []
X_time = []
y_func = []

for user_id, user_df in df.groupby('user_idx'):

    user_df = user_df.sort_values('t0')

    func_seq = user_df['func_idx'].values
    time_seq = user_df[['hour_s','minute_s','day_of_week_s','duration_s']].values

    for i in range(len(user_df) - sequence_length):

        X_func.append(func_seq[i:i+sequence_length])
        X_user.append([user_id]*sequence_length)
        X_time.append(time_seq[i:i+sequence_length])
        y_func.append(func_seq[i+sequence_length])

X_func = np.array(X_func)
X_user = np.array(X_user)
X_time = np.array(X_time)

y_func = to_categorical(
    y_func,
    num_classes=len(func_enc.classes_)
)

print("Sequences prepared:", X_func.shape, X_time.shape, y_func.shape)

# =========================
# 7️⃣ Build GRU model
# =========================
def build_sequence_model(num_funcs, num_users, seq_len, time_features=4):

    in_func = layers.Input(shape=(seq_len,), name='func_input')
    in_user = layers.Input(shape=(seq_len,), name='user_input')
    in_time = layers.Input(shape=(seq_len,time_features), name='time_input')

    emb_f = layers.Embedding(num_funcs, 64)(in_func)
    emb_u = layers.Embedding(num_users, 32)(in_user)

    x = layers.Concatenate()([emb_f,emb_u,in_time])

    x = layers.Bidirectional(
        layers.GRU(
            96,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1
        )
    )(x)

    x = layers.Bidirectional(
        layers.GRU(
            48,
            dropout=0.2,
            recurrent_dropout=0.1
        )
    )(x)

    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(num_funcs,activation='softmax')(x)

    model = Model(
        inputs=[in_func,in_user,in_time],
        outputs=out
    )

    optimizer = Adam(
        learning_rate=0.0005,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 🔹 MODEL CREATE HERE (important)
model = build_sequence_model(
    len(func_enc.classes_),
    len(user_enc.classes_),
    sequence_length
)

# =========================
# 8️⃣ Train model
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

history = model.fit(
    [X_func,X_user,X_time],
    y_func,
    batch_size=8,
    epochs=40,
    validation_split=0.2,
    callbacks=[early_stop,reduce_lr],
    shuffle=True
)

# =========================
# 9️⃣ Save model
# =========================
model.save('gru_next_function_model6.h5')

with open('func_encoder6.pkl','wb') as f:
    pickle.dump(func_enc,f)

with open('user_encoder6.pkl','wb') as f:
    pickle.dump(user_enc,f)

with open('time_scaler6.pkl','wb') as f:
    pickle.dump(scaler,f)

print("✅ GRU model saved.")
print("✅ Encoders and scaler saved.")