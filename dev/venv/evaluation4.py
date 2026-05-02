import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_recall_fscore_support, top_k_accuracy_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def extract_model_outputs(pred_output):
    if isinstance(pred_output, (list, tuple)):
        class_prob = np.array(pred_output[0])
        interval_pred = None
        if len(pred_output) > 1:
            interval_pred = np.array(pred_output[1]).reshape(-1)
        return class_prob, interval_pred
    return np.array(pred_output), None

# ==========================================
# 1. EXTENDED METRICS CALCULATION
# ==========================================
def calculate_comprehensive_metrics(y_true, y_pred, y_prob, func_enc):
    num_classes = len(func_enc.classes_)
    top3_k = min(3, num_classes)
    top5_k = min(5, num_classes)

    # Standard Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Top-K Accuracy (Crucial for "Next Step" Recommenders)
    top3_acc = top_k_accuracy_score(y_true, y_prob, k=top3_k, labels=np.arange(num_classes))
    top5_acc = top_k_accuracy_score(y_true, y_prob, k=top5_k, labels=np.arange(num_classes))
    
    # Macro vs Weighted F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print("\n" + "="*30)
    print("  MODEL EVALUATION SUMMARY")
    print("="*30)
    print(f"Standard Accuracy (Top-1): {acc:.4f}")
    print(f"Top-{top3_k} Accuracy:           {top3_acc:.4f}")
    print(f"Top-{top5_k} Accuracy:           {top5_acc:.4f}")
    print(f"Weighted F1-Score:        {f1:.4f}")
    print(f"Weighted Precision:       {precision:.4f}")
    print(f"Weighted Recall:          {recall:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(num_classes),
            target_names=func_enc.classes_,
            zero_division=0
        )
    )


def calculate_train_val_metrics(model, x_func, x_user, x_time, y_true, num_classes, y_interval=None):
    try:
        if y_interval is None:
            split = train_test_split(
                x_func, x_user, x_time, y_true, test_size=0.2, random_state=42, stratify=y_true
            )
            x_func_train, x_func_val, x_user_train, x_user_val, x_time_train, x_time_val, y_train, y_val = split
            y_int_train, y_int_val = None, None
        else:
            split = train_test_split(
                x_func, x_user, x_time, y_true, y_interval, test_size=0.2, random_state=42, stratify=y_true
            )
            (
                x_func_train, x_func_val,
                x_user_train, x_user_val,
                x_time_train, x_time_val,
                y_train, y_val,
                y_int_train, y_int_val,
            ) = split
    except ValueError:
        if y_interval is None:
            split = train_test_split(
                x_func, x_user, x_time, y_true, test_size=0.2, random_state=42, stratify=None
            )
            x_func_train, x_func_val, x_user_train, x_user_val, x_time_train, x_time_val, y_train, y_val = split
            y_int_train, y_int_val = None, None
        else:
            split = train_test_split(
                x_func, x_user, x_time, y_true, y_interval, test_size=0.2, random_state=42, stratify=None
            )
            (
                x_func_train, x_func_val,
                x_user_train, x_user_val,
                x_time_train, x_time_val,
                y_train, y_val,
                y_int_train, y_int_val,
            ) = split

    train_out = model.predict([x_func_train, x_user_train, x_time_train], verbose=0)
    val_out = model.predict([x_func_val, x_user_val, x_time_val], verbose=0)
    train_prob, train_interval_pred = extract_model_outputs(train_out)
    val_prob, val_interval_pred = extract_model_outputs(val_out)

    train_pred = np.argmax(train_prob, axis=1)
    val_pred = np.argmax(val_prob, axis=1)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    mse = mean_squared_error(y_val, val_pred)
    mae = mean_absolute_error(y_val, val_pred)

    print("\n" + "=" * 30)
    print("  TRAIN / VALIDATION METRICS")
    print("=" * 30)
    print(f"Train Accuracy:           {train_acc:.4f}")
    print(f"Train MSE:                {train_mse:.4f}")
    print(f"Train MAE:                {train_mae:.4f}")
    print(f"Validation Accuracy:      {val_acc:.4f}")
    print(f"Validation MSE:           {mse:.4f}")
    print(f"Validation MAE:           {mae:.4f}")

    if y_int_train is not None and train_interval_pred is not None:
        int_train_mse = mean_squared_error(y_int_train, train_interval_pred)
        int_train_mae = mean_absolute_error(y_int_train, train_interval_pred)
        int_val_mse = mean_squared_error(y_int_val, val_interval_pred)
        int_val_mae = mean_absolute_error(y_int_val, val_interval_pred)
        print("-" * 30)
        print("Interval (Scaled) Metrics:")
        print(f"Train Interval MSE:       {int_train_mse:.4f}")
        print(f"Train Interval MAE:       {int_train_mae:.4f}")
        print(f"Val Interval MSE:         {int_val_mse:.4f}")
        print(f"Val Interval MAE:         {int_val_mae:.4f}")

    print("-" * 30)

# ==========================================
# 2. VISUALIZATION SUITE
# ==========================================
def plot_evaluation_graphs(y_true, y_pred, y_prob, func_enc):
    num_classes = len(func_enc.classes_)
    top3_k = min(3, num_classes)
    top5_k = min(5, num_classes)

    # Set visual style
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(20, 15))
    
    # --- Plot 1: Top-K Accuracy Bar Chart ---
    ax1 = plt.subplot(2, 2, 1)
    k_values = ['Top-1', f'Top-{top3_k}', f'Top-{top5_k}']
    k_scores = [
        accuracy_score(y_true, y_pred),
        top_k_accuracy_score(y_true, y_prob, k=top3_k, labels=np.arange(num_classes)),
        top_k_accuracy_score(y_true, y_prob, k=top5_k, labels=np.arange(num_classes))
    ]
    sns.barplot(x=k_values, y=k_scores, ax=ax1, palette="viridis")
    ax1.set_title("Model Accuracy at Different K-Levels", fontsize=14)
    ax1.set_ylim(0, 1.1)
    for i, v in enumerate(k_scores):
        ax1.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')

    # --- Plot 2: Confusion Matrix Heatmap ---
    ax2 = plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    # Normalize by row to show percentages
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=func_enc.classes_, yticklabels=func_enc.classes_)
    ax2.set_title("Normalized Confusion Matrix", fontsize=14)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    # --- Plot 3: Precision-Recall per Class ---
    ax3 = plt.subplot(2, 2, 3)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=func_enc.classes_,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).T.iloc[:num_classes]
    report_df[['precision', 'recall', 'f1-score']].plot(kind='barh', ax=ax3)
    ax3.set_title("Metrics per Function Class", fontsize=14)
    ax3.legend(loc='lower right')

    # --- Plot 4: Prediction Confidence Distribution ---
    ax4 = plt.subplot(2, 2, 4)
    confidences = np.max(y_prob, axis=1)
    correct_mask = (y_true == y_pred)
    
    sns.kdeplot(confidences[correct_mask], label='Correct Preds', fill=True, ax=ax4, color='green')
    sns.kdeplot(confidences[~correct_mask], label='Incorrect Preds', fill=True, ax=ax4, color='red')
    ax4.set_title("Confidence Distribution (Correct vs Incorrect)", fontsize=14)
    ax4.set_xlabel("Model Confidence (Probability)")
    ax4.legend()

    plt.tight_layout()
    plt.show()

# ==========================================
# 3. EXECUTION
# ==========================================
def prepare_evaluation_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "gru_dataset_fixed_dates_8000.csv")
    model_path = os.path.join(base_dir, "gru_neural_croston_model11.h5")
    interval_scaler_path = os.path.join(base_dir, "interval_scaler.pkl")

    with open(interval_scaler_path, "rb") as f:
        interval_scaler = pickle.load(f)

    model = tf.keras.models.load_model(model_path, compile=False)
    embedding_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Embedding)]
    if len(embedding_layers) < 3:
        raise ValueError("Could not infer embedding dimensions from model.")
    max_func_idx = embedding_layers[0].input_dim - 1
    max_user_idx = embedding_layers[1].input_dim - 1
    max_time_idx = embedding_layers[2].input_dim - 1
    sequence_len = int(model.input_shape[0][1])
    df = pd.read_csv(dataset_path)

    func_enc = LabelEncoder()
    user_enc = LabelEncoder()
    time_enc = LabelEncoder()

    df['func_idx'] = func_enc.fit_transform(df['function'].astype(str))
    df['user_idx'] = user_enc.fit_transform(df['user'].astype(str))

    normalized_hour = (
        df['hour']
        .astype(str)
        .str.strip()
        .str.replace('.', ':', regex=False)
    )
    df['hour_dt'] = pd.to_datetime(normalized_hour, format='%I:%M:%S %p', errors='coerce')
    if df['hour_dt'].isna().any():
        bad_values = normalized_hour[df['hour_dt'].isna()].head(5).tolist()
        raise ValueError(f"Unparseable 'hour' values found, examples: {bad_values}")

    df['time_idx'] = time_enc.fit_transform(df['hour_dt'].dt.strftime('%H:%M:%S'))

    if int(df['func_idx'].max()) > max_func_idx:
        raise ValueError(
            f"Function index exceeds model embedding range: max {int(df['func_idx'].max())}, allowed {max_func_idx}."
        )
    if int(df['user_idx'].max()) > max_user_idx:
        raise ValueError(
            f"User index exceeds model embedding range: max {int(df['user_idx'].max())}, allowed {max_user_idx}."
        )
    if int(df['time_idx'].max()) > max_time_idx:
        overflow = int((df['time_idx'] > max_time_idx).sum())
        print(f"[Info] Clipping {overflow} time indices to max supported value {max_time_idx}.")
        df['time_idx'] = np.minimum(df['time_idx'], max_time_idx)

    df['interval'] = df.groupby('user')['t0'].diff().fillna(0)
    df['interval_scaled'] = interval_scaler.transform(df[['interval']])

    x_func, x_user, x_time, y_true, y_interval_true = [], [], [], [], []
    for user_id, user_df in df.groupby('user_idx'):
        user_df = user_df.sort_values('t0')
        func_seq = user_df['func_idx'].values
        time_seq = user_df['time_idx'].values
        interval_seq = user_df['interval_scaled'].values.reshape(-1)

        for i in range(len(user_df) - sequence_len):
            x_func.append(func_seq[i:i + sequence_len])
            x_user.append([user_id] * sequence_len)
            x_time.append(time_seq[i:i + sequence_len])
            y_true.append(func_seq[i + sequence_len])
            y_interval_true.append(interval_seq[i + sequence_len])

    if not y_true:
        raise ValueError("No evaluation sequences were created. Check dataset size and sequence length.")

    x_func = np.array(x_func)
    x_user = np.array(x_user)
    x_time = np.array(x_time)
    y_true = np.array(y_true)
    y_interval_true = np.array(y_interval_true)

    pred_out = model.predict([x_func, x_user, x_time], verbose=0)
    y_prob, y_interval_pred = extract_model_outputs(pred_out)
    y_pred = np.argmax(y_prob, axis=1)
    return model, x_func, x_user, x_time, y_true, y_interval_true, y_interval_pred, y_pred, y_prob, func_enc


if __name__ == "__main__":
    model, x_func, x_user, x_time, y_true, y_interval_true, y_interval_pred, y_pred, y_prob, func_enc = prepare_evaluation_data()
    calculate_train_val_metrics(model, x_func, x_user, x_time, y_true, len(func_enc.classes_), y_interval_true)
    calculate_comprehensive_metrics(y_true, y_pred, y_prob, func_enc)
    if y_interval_pred is not None:
        print("\n" + "=" * 30)
        print("  FULL DATA INTERVAL METRICS")
        print("=" * 30)
        print(f"Interval MSE (scaled):    {mean_squared_error(y_interval_true, y_interval_pred):.4f}")
        print(f"Interval MAE (scaled):    {mean_absolute_error(y_interval_true, y_interval_pred):.4f}")
        print("-" * 30)
    plot_evaluation_graphs(y_true, y_pred, y_prob, func_enc)
