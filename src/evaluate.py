"""
evaluate.py

Evaluation & visualization for RetNet vs Transformer collision prediction.
Generates all plots needed for your Review 2 report.

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint checkpoints/best_retnet.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup for clean plots
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

from model import RetNetForCollisionPrediction, TransformerBaseline


# ─── Load model from checkpoint ──────────────────────────────────

def load_model_from_checkpoint(ckpt_path, device):
    """Load model from saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "retnet")
    input_dim = ckpt.get("input_dim", 14)

    if model_name == "retnet":
        model = RetNetForCollisionPrediction(input_dim=input_dim)
    else:
        model = TransformerBaseline(input_dim=input_dim)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


# ─── Get predictions ─────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    """Return arrays of (labels, probs, logits)."""
    all_labels = []
    all_probs = []
    all_logits = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        probs = torch.sigmoid(logits)

        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y_batch.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.concatenate(all_logits),
    )


# ─── Plotting functions ──────────────────────────────────────────

def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix — {model_name.upper()}')
    plt.colorbar(im, ax=ax)

    labels = ['Low-risk\n(Pc ≤ 10⁻⁴)', 'High-risk\n(Pc > 10⁻⁴)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curve(labels, probs, model_name, save_path):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'{model_name.upper()} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {model_name.upper()}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pr_curve(labels, probs, model_name, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, 'r-', lw=2,
            label=f'{model_name.upper()} (AP = {ap:.4f})')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {model_name.upper()}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_history(history, model_name, save_path):
    """Plot training/validation loss and F1 over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training & Validation Loss — {model_name.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1
    ax2.plot(epochs, history['val_f1'], 'g-', label='Val F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'Validation F1 — {model_name.upper()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probability_distribution(labels, probs, model_name, save_path):
    """Plot histogram of predicted probabilities, separated by true class."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    ax.hist(neg_probs, bins=50, alpha=0.6, color='blue', label=f'Low-risk (n={len(neg_probs)})')
    ax.hist(pos_probs, bins=50, alpha=0.6, color='red', label=f'High-risk (n={len(pos_probs)})')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Decision threshold')
    ax.set_xlabel('Predicted Collision Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'Probability Distribution — {model_name.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_table(retnet_metrics, transformer_metrics, save_path):
    """Plot side-by-side comparison table of both models."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    retnet_vals = [retnet_metrics['accuracy'], retnet_metrics['precision'],
                   retnet_metrics['recall'], retnet_metrics['f1'],
                   retnet_metrics['auc_roc']]
    trans_vals = [transformer_metrics['accuracy'], transformer_metrics['precision'],
                  transformer_metrics['recall'], transformer_metrics['f1'],
                  transformer_metrics['auc_roc']]

    # Build table
    cell_text = []
    for i in range(len(metrics_names)):
        r_val = f'{retnet_vals[i]:.4f}'
        t_val = f'{trans_vals[i]:.4f}'
        winner = 'RetNet' if retnet_vals[i] > trans_vals[i] else 'Transformer'
        cell_text.append([metrics_names[i], r_val, t_val, winner])

    table = ax.table(
        cellText=cell_text,
        colLabels=['Metric', 'RetNet', 'Transformer', 'Winner'],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Color header
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight winners
    for i in range(1, len(metrics_names) + 1):
        if cell_text[i-1][3] == 'RetNet':
            table[i, 1].set_facecolor('#C6EFCE')
        else:
            table[i, 2].set_facecolor('#C6EFCE')

    ax.set_title('RetNet vs Transformer — Test Set Comparison', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Main evaluation ─────────────────────────────────────────────

def evaluate_model(ckpt_path, data_dir, output_dir, device_str='auto'):
    """Full evaluation pipeline for one model."""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    # Load model
    model, model_name = load_model_from_checkpoint(ckpt_path, device)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name.upper()}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load data
    X_test, y_test = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    # Get predictions
    labels, probs, logits = get_predictions(model, test_loader, device)
    preds = (probs > 0.5).astype(int)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'auc_roc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        'avg_precision': average_precision_score(labels, probs),
        'confusion_matrix': confusion_matrix(labels, preds),
    }

    # Print metrics
    print(f"\n  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(labels, preds, target_names=['Low-risk', 'High-risk']))

    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    prefix = model_name.lower()

    plot_confusion_matrix(
        metrics['confusion_matrix'], model_name,
        os.path.join(output_dir, f'{prefix}_confusion_matrix.png'))

    plot_roc_curve(
        labels, probs, model_name,
        os.path.join(output_dir, f'{prefix}_roc_curve.png'))

    plot_pr_curve(
        labels, probs, model_name,
        os.path.join(output_dir, f'{prefix}_pr_curve.png'))

    plot_probability_distribution(
        labels, probs, model_name,
        os.path.join(output_dir, f'{prefix}_prob_distribution.png'))

    return metrics


# ─── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_retnet.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.data_dir, args.output_dir, args.device)
