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
from tensorflow.keras.utils import to_categorical

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


def calculate_train_val_metrics(model, x_func, x_user, x_time, x_status, y_true, num_classes):
    try:
        split = train_test_split(
            x_func, x_user, x_time, x_status, y_true, test_size=0.2, random_state=42, stratify=y_true
        )
    except ValueError:
        split = train_test_split(
            x_func, x_user, x_time, x_status, y_true, test_size=0.2, random_state=42, stratify=None
        )

    (
        x_func_train, x_func_val,
        x_user_train, x_user_val,
        x_time_train, x_time_val,
        x_status_train, x_status_val,
        y_train, y_val,
    ) = split

    y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=num_classes)

    train_loss, train_acc = model.evaluate(
        [x_func_train, x_user_train, x_time_train, x_status_train], y_train_one_hot, verbose=0
    )
    val_loss, val_acc = model.evaluate(
        [x_func_val, x_user_val, x_time_val, x_status_val], y_val_one_hot, verbose=0
    )

    val_prob = model.predict([x_func_val, x_user_val, x_time_val, x_status_val], verbose=0)
    val_pred = np.argmax(val_prob, axis=1)
    mse = mean_squared_error(y_val, val_pred)
    mae = mean_absolute_error(y_val, val_pred)

    print("\n" + "=" * 30)
    print("  TRAIN / VALIDATION METRICS")
    print("=" * 30)
    print(f"Train Loss:               {train_loss:.4f}")
    print(f"Train Accuracy:           {train_acc:.4f}")
    print(f"Validation Loss:          {val_loss:.4f}")
    print(f"Validation Accuracy:      {val_acc:.4f}")
    print(f"Validation MSE:           {mse:.4f}")
    print(f"Validation MAE:           {mae:.4f}")
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
    model_path = os.path.join(base_dir, "gru_focal_loss_model10copy2.h5")
    func_enc_path = os.path.join(base_dir, "func_encoder10copy2.pkl")
    user_enc_path = os.path.join(base_dir, "user_encoder10copy2.pkl")
    time_enc_path = os.path.join(base_dir, "time_encoder10copy2.pkl")

    with open(func_enc_path, "rb") as f:
        func_enc = pickle.load(f)
    with open(user_enc_path, "rb") as f:
        user_enc = pickle.load(f)
    with open(time_enc_path, "rb") as f:
        time_enc = pickle.load(f)

    model = tf.keras.models.load_model(model_path)
    sequence_len = int(model.input_shape[0][1])
    df = pd.read_csv(dataset_path)

    df['func_idx'] = func_enc.transform(df['function'].astype(str))
    df['user_idx'] = user_enc.transform(df['user'].astype(str))

    normalized_hour = (
        df['hour']
        .astype(str)
        .str.strip()
        .str.replace('.', ':', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
    )
    df['hour_dt'] = pd.to_datetime(normalized_hour, format='%I:%M:%S %p', errors='coerce')
    if df['hour_dt'].isna().any():
        bad_values = normalized_hour[df['hour_dt'].isna()].head(5).tolist()
        raise ValueError(f"Unparseable 'hour' values found, examples: {bad_values}")

    # Some evaluation timestamps may not exist in the fitted LabelEncoder classes.
    # Map each timestamp to the nearest known time-of-day class instead of failing.
    time_labels = df['hour_dt'].dt.strftime('%H:%M:%S')
    known_labels = np.asarray(time_enc.classes_, dtype=str)

    input_seconds = (
        df['hour_dt'].dt.hour * 3600
        + df['hour_dt'].dt.minute * 60
        + df['hour_dt'].dt.second
    ).to_numpy(dtype=np.int32)

    known_times = pd.to_datetime(known_labels, format='%H:%M:%S')
    known_seconds = (
        known_times.hour * 3600
        + known_times.minute * 60
        + known_times.second
    ).to_numpy(dtype=np.int32)

    insert_pos = np.searchsorted(known_seconds, input_seconds)
    left_idx = np.clip(insert_pos - 1, 0, len(known_seconds) - 1)
    right_idx = np.clip(insert_pos, 0, len(known_seconds) - 1)

    left_dist = np.abs(input_seconds - known_seconds[left_idx])
    right_dist = np.abs(known_seconds[right_idx] - input_seconds)
    nearest_idx = np.where(right_dist < left_dist, right_idx, left_idx)

    unseen_mask = ~time_labels.isin(known_labels)
    if unseen_mask.any():
        unseen_count = int(unseen_mask.sum())
        print(f"[Info] Mapped {unseen_count} unseen time labels to nearest known classes.")

    df['time_idx'] = nearest_idx

    df['status_idx'] = pd.to_numeric(df['status'], errors='coerce').fillna(0).clip(0, 1).astype(np.int32)

    x_func, x_user, x_time, x_status, y_true = [], [], [], [], []
    for user_id, user_df in df.groupby('user_idx'):
        user_df = user_df.sort_values('t0')
        func_seq = user_df['func_idx'].values
        time_seq = user_df['time_idx'].values
        status_seq = user_df['status_idx'].values

        for i in range(len(user_df) - sequence_len):
            x_func.append(func_seq[i:i + sequence_len])
            x_user.append([user_id] * sequence_len)
            x_time.append(time_seq[i:i + sequence_len])
            x_status.append(status_seq[i:i + sequence_len])
            y_true.append(func_seq[i + sequence_len])

    if not y_true:
        raise ValueError("No evaluation sequences were created. Check dataset size and sequence length.")

    x_func = np.array(x_func)
    x_user = np.array(x_user)
    x_time = np.array(x_time)
    x_status = np.array(x_status, dtype=np.float32).reshape(-1, sequence_len, 1)
    y_true = np.array(y_true)

    y_prob = model.predict([x_func, x_user, x_time, x_status], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return model, x_func, x_user, x_time, x_status, y_true, y_pred, y_prob, func_enc


if __name__ == "__main__":
    model, x_func, x_user, x_time, x_status, y_true, y_pred, y_prob, func_enc = prepare_evaluation_data()
    calculate_train_val_metrics(model, x_func, x_user, x_time, x_status, y_true, len(func_enc.classes_))
    calculate_comprehensive_metrics(y_true, y_pred, y_prob, func_enc)
    plot_evaluation_graphs(y_true, y_pred, y_prob, func_enc)
