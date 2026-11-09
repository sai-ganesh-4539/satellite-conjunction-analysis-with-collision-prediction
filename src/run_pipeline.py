"""
run_full_pipeline.py — Train both models, evaluate, generate all comparison plots.
One script to run everything end-to-end.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Path Resolution ─────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == 'src':
    REPO_ROOT = os.path.dirname(current_dir)
else:
    REPO_ROOT = current_dir

sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from model import RetNetForCollisionPrediction, TransformerBaseline
from config import *

# ─── Derived Config ──────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WARMUP_EPOCHS = max(1, int(EPOCHS * WARMUP_RATIO))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')


# ─── Focal Loss (Numerically Stable with Logits) ─────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits = logits.squeeze(-1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_weight * focal_weight * bce).mean()


# ─── Train one epoch ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


# ─── Evaluate ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        if criterion is not None:
            loss = criterion(logits, y_batch)
        else:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), y_batch.float())
        total_loss += loss.item() * X_batch.size(0)
        all_probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)
    n_samples = len(loader.dataset) if hasattr(loader, 'dataset') else sum(len(x[0]) for x in loader)
    metrics = {
        'loss': total_loss / max(1, n_samples),
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'auc_roc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
    }
    return metrics, probs, labels


# ─── Full training loop ──────────────────────────────────────────
def train_model(model, name, train_loader, val_loader, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    # Warmup + Cosine schedule
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    best_f1 = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}

    print(f"\nTraining {name.upper()} for {EPOCHS} epochs on {device}...")
    print(f"  LR: {LR}, Warmup: {WARMUP_EPOCHS} epochs, Patience: {PATIENCE}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_metrics, _, _ = evaluate(model, val_loader, device, criterion=criterion)
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['f1']
        val_auc = val_metrics['auc_roc']

        elapsed = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:3d}/{EPOCHS} | "
              f"LR: {current_lr:.2e} | "
              f"TL: {train_loss:.4f} | VL: {val_loss:.4f} | "
              f"F1: {val_f1:.4f} | AUC: {val_auc:.4f} | "
              f"{elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{name}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'model_name': name,
                'input_dim': model.input_proj.in_features,
            }, ckpt_path)
            print(f"  *** Best F1: {val_f1:.4f} (saved) ***")

        if epoch - best_epoch >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("-" * 70)
    print(f"Done. Best epoch: {best_epoch}, Best F1: {best_f1:.4f}")
    return history


# ─── Plotting ────────────────────────────────────────────────────
def plot_all(retnet_hist, trans_hist, r_metrics, t_metrics,
             r_probs, t_probs, test_labels, r_cm, t_cm):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Comparison table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    rows = []
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        rv, tv = r_metrics[k], t_metrics[k]
        winner = 'RetNet' if rv > tv else ('Transformer' if tv > rv else 'Tie')
        rows.append([k.title(), f'{rv:.4f}', f'{tv:.4f}', winner])
    table = ax.table(cellText=rows,
                     colLabels=['Metric', 'RetNet', 'Transformer', 'Winner'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        if rows[i-1][3] == 'RetNet':
            table[i, 1].set_facecolor('#C6EFCE')
        elif rows[i-1][3] == 'Transformer':
            table[i, 2].set_facecolor('#C6EFCE')
    ax.set_title('RetNet vs Transformer — Test Set Comparison', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_table.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: results/comparison_table.png")

    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    fpr1, tpr1, _ = roc_curve(test_labels, r_probs)
    fpr2, tpr2, _ = roc_curve(test_labels, t_probs)
    ax.plot(fpr1, tpr1, 'b-', lw=2.5, label=f'RetNet (AUC = {r_metrics["auc_roc"]:.4f})')
    ax.plot(fpr2, tpr2, 'r-', lw=2.5, label=f'Transformer (AUC = {t_metrics["auc_roc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — RetNet vs Transformer', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_comparison.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: results/roc_comparison.png")

    # 3. Precision-Recall curves
    pr1, rc1, _ = precision_recall_curve(test_labels, r_probs)
    pr2, rc2, _ = precision_recall_curve(test_labels, t_probs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(rc1, pr1, 'b-', lw=2.5, label=f'RetNet (AP = {average_precision_score(test_labels, r_probs):.4f})')
    ax.plot(rc2, pr2, 'r-', lw=2.5, label=f'Transformer (AP = {average_precision_score(test_labels, t_probs):.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve — RetNet vs Transformer', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pr_comparison.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: results/pr_comparison.png")

    # 4. Confusion matrices
    for cm, name, color in [(r_cm, 'RetNet', 'Blues'), (t_cm, 'Transformer', 'Reds')]:
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap(color)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(f'Confusion Matrix — {name}', fontsize=13)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Low-risk', 'High-risk'])
        ax.set_yticklabels(['Low-risk', 'High-risk'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        thresh = cm.max() / 2 if cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=18, fontweight='bold',
                        color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{name.lower()}_confusion_matrix.png'), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: results/{name.lower()}_confusion_matrix.png")

    # 5. Training curves
    for hist, name, color in [(retnet_hist, 'RetNet', 'blue'), (trans_hist, 'Transformer', 'red')]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(hist['train_loss']) + 1)
        ax1.plot(epochs, hist['train_loss'], color=color, label='Train Loss')
        ax1.plot(epochs, hist['val_loss'], color=color, linestyle='--', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{name} — Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, hist['val_f1'], color=color, label='Val F1')
        ax2.plot(epochs, hist['val_auc'], color=color, linestyle='--', label='Val AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title(f'{name} — Validation F1 & AUC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{name.lower()}_training_curves.png'), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: results/{name.lower()}_training_curves.png")

    # 6. Probability distributions
    for probs, name, color in [(r_probs, 'RetNet', 'blue'), (t_probs, 'Transformer', 'red')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = probs[test_labels == 1]
        neg = probs[test_labels == 0]
        ax.hist(neg, bins=25, alpha=0.6, color=color, label=f'Low-risk (n={len(neg)})')
        ax.hist(pos, bins=25, alpha=0.6, color='orange', label=f'High-risk (n={len(pos)})')
        ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        ax.set_xlabel('Predicted Collision Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} — Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{name.lower()}_prob_distribution.png'), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: results/{name.lower()}_prob_distribution.png")


# ─── Main ────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, "
          f"lr={LR}, epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"  gammas={GAMMAS}, focal_alpha={FOCAL_ALPHA}, patience={PATIENCE}")

    # Load data
    train_path = os.path.join(PROCESSED_DIR, 'train.pt')
    val_path = os.path.join(PROCESSED_DIR, 'val.pt')
    test_path = os.path.join(PROCESSED_DIR, 'test.pt')

    if not os.path.exists(train_path):
        print(f"Processed data not found at {PROCESSED_DIR}. Running data_processing.py ...")
        from data_processing import build_dataset
        build_dataset(output_dir=PROCESSED_DIR)

    X_train, y_train = torch.load(train_path, weights_only=False)
    X_val, y_val = torch.load(val_path, weights_only=False)
    X_test, y_test = torch.load(test_path, weights_only=False)
    input_dim = X_train.shape[-1]

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Input dim:      {input_dim}")
    print(f"  Sequence len:   {X_train.shape[1]}")
    print(f"  Train samples:  {len(X_train)} (high-risk: {int(y_train.sum())}, low-risk: {len(y_train)-int(y_train.sum())})")
    print(f"  Val samples:    {len(X_val)} (high-risk: {int(y_val.sum())}, low-risk: {len(y_val)-int(y_val.sum())})")
    print(f"  Test samples:   {len(X_test)} (high-risk: {int(y_test.sum())}, low-risk: {len(y_test)-int(y_test.sum())})")
    print(f"  Class ratio:    {int(y_train.sum())}:{int(len(y_train)-y_train.sum())} (pos:neg)")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Train RetNet
    print(f"\n{'='*60}")
    print(f"MODEL 1: RetNet")
    print(f"{'='*60}")
    retnet = RetNetForCollisionPrediction(
        input_dim=input_dim, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, gammas=GAMMAS, dropout=DROPOUT, pooling=POOLING
    )
    print(f"Parameters: {sum(p.numel() for p in retnet.parameters()):,}")
    retnet_hist = train_model(retnet, 'retnet', train_loader, val_loader, DEVICE)

    # Train Transformer
    print(f"\n{'='*60}")
    print(f"MODEL 2: Transformer Baseline")
    print(f"{'='*60}")
    transformer = TransformerBaseline(
        input_dim=input_dim, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT, pooling=POOLING
    )
    print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    trans_hist = train_model(transformer, 'transformer', train_loader, val_loader, DEVICE)

    # ─── Test evaluation ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")

    # Load best checkpoints
    r_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_retnet.pt'), map_location=DEVICE, weights_only=False)
    retnet = RetNetForCollisionPrediction(
        input_dim=input_dim, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, gammas=GAMMAS, dropout=DROPOUT, pooling=POOLING
    ).to(DEVICE)
    retnet.load_state_dict(r_ckpt['model_state_dict'])

    t_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_transformer.pt'), map_location=DEVICE, weights_only=False)
    transformer = TransformerBaseline(
        input_dim=input_dim, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT, pooling=POOLING
    ).to(DEVICE)
    transformer.load_state_dict(t_ckpt['model_state_dict'])

    test_crit = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    r_metrics, r_probs, r_labels = evaluate(retnet, test_loader, DEVICE, criterion=test_crit)
    t_metrics, t_probs, t_labels = evaluate(transformer, test_loader, DEVICE, criterion=test_crit)

    r_preds = (r_probs > 0.5).astype(int)
    t_preds = (t_probs > 0.5).astype(int)
    r_cm = confusion_matrix(r_labels, r_preds)
    t_cm = confusion_matrix(t_labels, t_preds)

    print(f"\n--- RetNet ---")
    for k, v in r_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Confusion Matrix:\n{r_cm}")

    print(f"\n--- Transformer ---")
    for k, v in t_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Confusion Matrix:\n{t_cm}")

    # ─── Generate all plots ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS")
    print(f"{'='*60}")
    plot_all(retnet_hist, trans_hist, r_metrics, t_metrics,
             r_probs, t_probs, r_labels, r_cm, t_cm)

    # ─── Save metrics to file ────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("RetNet vs Transformer - Test Set Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Metric':<15} {'RetNet':>10} {'Transformer':>12} {'Winner':>10}\n")
        f.write("-" * 50 + "\n")
        for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            rv, tv = r_metrics[k], t_metrics[k]
            w = 'RetNet' if rv > tv else ('Transformer' if tv > rv else 'Tie')
            f.write(f"{k:<15} {rv:>10.4f} {tv:>12.4f} {w:>10}\n")
        f.write(f"\nRetNet Confusion Matrix:\n{r_cm}\n")
        f.write(f"Transformer Confusion Matrix:\n{t_cm}\n")
    print(f"Saved: {metrics_path}")

    # ─── Final comparison ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON: RetNet vs Transformer")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'RetNet':>10} {'Transformer':>12} {'Winner':>10}")
    print("-" * 50)
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        rv, tv = r_metrics[k], t_metrics[k]
        w = 'RetNet' if rv > tv else ('Transformer' if tv > rv else 'Tie')
        print(f"{k:<15} {rv:>10.4f} {tv:>12.4f} {w:>10}")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
