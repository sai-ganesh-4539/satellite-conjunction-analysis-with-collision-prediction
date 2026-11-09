 """
train.py

Training loop for RetNet vs Transformer baseline on satellite collision prediction.

Features:
    - Focal Loss (handles class imbalance ~5% high-risk)
    - AdamW optimizer
    - Cosine annealing with warmup LR schedule
    - Early stopping on validation loss
    - Evaluation metrics: Accuracy, Precision, Recall, F1, AUC-ROC
    - Saves best model checkpoint

Usage:
    python src/train.py
    python src/train.py --model transformer --epochs 50
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from model import (
    RetNetForCollisionPrediction,
    TransformerBaseline,
    count_parameters,
)


# ─── Focal Loss ───────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Default gamma=2.0 focuses on hard examples.
    alpha balances positive/negative classes.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 1) raw logits (NOT sigmoid'd)
            targets: (batch,) binary labels {0, 1}
        """
        probs = torch.sigmoid(logits).squeeze(-1)  # (batch,)
        targets = targets.float()

        # Binary cross-entropy
        bce = nn.functional.binary_cross_entropy(probs, targets, reduction="none")

        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ─── Cosine Annealing with Warmup ────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """
    Create a cosine schedule with linear warmup.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─── Evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """
    Run evaluation and return metrics dict.
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs > 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
    }

    # AUC-ROC (only if both classes present)
    if len(np.unique(all_labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc_roc"] = 0.0

    metrics["confusion_matrix"] = confusion_matrix(all_labels, preds).tolist()
    return metrics


# ─── Training Loop ────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * X_batch.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(loader.dataset)


def train(
    model_name: str = "retnet",
    data_dir: str = "data/processed",
    output_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_ratio: float = 0.1,
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    patience: int = 15,
    device_str: str = "auto",
):
    """
    Full training pipeline.
    """
    # --- Device ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # --- Load data ---
    print("\nLoading processed data...")
    X_train, y_train = torch.load(os.path.join(data_dir, "train.pt"))
    X_val, y_val = torch.load(os.path.join(data_dir, "val.pt"))
    X_test, y_test = torch.load(os.path.join(data_dir, "test.pt"))

    input_dim = X_train.shape[-1]
    print(f"  Input dim: {input_dim}, Seq len: {X_train.shape[1]}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # --- Model ---
    if model_name == "retnet":
        model = RetNetForCollisionPrediction(
            input_dim=input_dim,
            d_model=128,
            n_heads=4,
            n_layers=4,
        ).to(device)
    elif model_name == "transformer":
        model = TransformerBaseline(
            input_dim=input_dim,
            d_model=128,
            n_heads=4,
            n_layers=4,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\nModel: {model_name}")
    print(f"  Parameters: {count_parameters(model):,}")

    # --- Loss, Optimizer, Scheduler ---
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"  Optimizer: AdamW (lr={lr}, wd={weight_decay})")
    print(f"  Scheduler: Cosine with {warmup_steps} warmup steps")
    print(f"  Loss: Focal (alpha={focal_alpha}, gamma={focal_gamma})")

    # --- Training ---
    os.makedirs(output_dir, exist_ok=True)
    best_val_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]

        epoch_time = time.time() - start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # Print metrics
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val AUC: {val_metrics['auc_roc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "model_name": model_name,
                "input_dim": input_dim,
            }, os.path.join(output_dir, f"best_{model_name}.pt"))
            print(f"  *** New best F1: {val_f1:.4f} (saved) ***")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    print("-" * 60)
    print(f"Training complete. Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")

    # --- Test evaluation ---
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Load best checkpoint
    ckpt = torch.load(os.path.join(output_dir, f"best_{model_name}.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print(f"  Test Loss:      {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  Test F1:        {test_metrics['f1']:.4f}")
    print(f"  Test AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")

    return history, test_metrics


# ─── Argument Parser ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RetNet/Transformer for collision prediction")
    parser.add_argument("--model", type=str, default="retnet", choices=["retnet", "transformer"],
                        help="Model to train")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
