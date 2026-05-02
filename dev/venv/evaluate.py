import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error
import tensorflow as tf

# Load Model and Data
model = tf.keras.models.load_model('intent_gru_model.h5', compile=False)
model.compile(optimizer='adam', loss='mean_absolute_error')
df = pd.read_csv('openfaas_research_data.csv') # Use same encoding/scaling logic as train_model.py
# (Note: In a real script, ensure you use the same LabelEncoder instance or save them)

# Taking the Evaluation set (Last 2500 rows)
eval_df = df.iloc[7500:].copy()
# (Assume preprocessing is done similar to train_model.py)
# Preprocessing evaluation data...
from sklearn.preprocessing import LabelEncoder, StandardScaler
f_enc, u_enc = LabelEncoder(), LabelEncoder()
eval_df['func_idx'] = f_enc.fit_transform(eval_df['function'])
eval_df['user_idx'] = u_enc.fit_transform(eval_df['user'])
eval_df[['hour_s', 'day_s']] = StandardScaler().fit_transform(eval_df[['hour', 'day']])

y_true = eval_df['derived_cold_start'].values
y_pred = model.predict([eval_df['func_idx'], eval_df['user_idx'], eval_df[['hour_s', 'day_s']]]).flatten()

# Metrics
print("\n--- RESEARCH EVALUATION METRICS ---")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_true + 1, y_pred + 1):.4f}")
print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")
print(f"MSLE: {mean_squared_log_error(y_true, np.clip(y_pred, 0, None)):.4f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_true, y_pred, alpha=0.3, color='blue', label='Predictions')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
plt.title('Intent-Aware GRU: Actual vs Predicted Cold Start')
plt.xlabel('Actual Cold Start (ms)')
plt.ylabel('Predicted Cold Start (ms)')
plt.legend()
plt.savefig('research_evaluation.png')
	## plt.show() removed to avoid blocking script execution