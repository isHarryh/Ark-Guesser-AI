"""Training pipeline for ArkGuesserModelV2 with improved training strategies.

Improvements over src/train.py:
  - CosineAnnealingWarmRestarts LR scheduler
  - Label smoothing for better calibration
  - Gradient clipping for training stability
  - Smarter early stopping
"""

import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from matplotlib import ticker, axes

from src.dataset import RawArkGuesserDataset, AugArkGuesserDataset
from src.model_v2 import ArkGuesserModelV2

TrainingRecord = namedtuple("TrainingRecord", ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_norm: float = 1.0,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def visualize_records(records: list[TrainingRecord], output_path: str = "outputs/training_metrics_v2.png"):
    if not records:
        return

    epochs = [r.epoch for r in records]
    train_losses = [r.train_loss for r in records]
    valid_losses = [r.valid_loss for r in records]
    train_accs = [r.train_acc for r in records]
    valid_accs = [r.valid_acc for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    assert isinstance(ax1, axes.Axes) and isinstance(ax2, axes.Axes)

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: round(x)))
    ax1.plot(epochs, train_losses, label="Train Loss", color="blue", marker="x")
    ax1.plot(epochs, valid_losses, label="Valid Loss", color="red", marker="*")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.4f}"))
    ax1.legend(loc="upper left")
    ax1.set_title("Loss (V2)")

    best_valid_loss_epoch = min(range(len(records)), key=lambda i: records[i].valid_loss) + 1
    best_valid_loss_record = records[best_valid_loss_epoch - 1]
    ax1.annotate(
        f"{best_valid_loss_record.valid_loss:.4f}",
        xy=(best_valid_loss_epoch, best_valid_loss_record.valid_loss),
        xytext=(best_valid_loss_epoch + 2, best_valid_loss_record.valid_loss),
        arrowprops=dict(arrowstyle="->"),
    )

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: round(x)))
    ax2.plot(epochs, train_accs, label="Train Accuracy", color="blue", marker="x", linestyle="--")
    ax2.plot(epochs, valid_accs, label="Valid Accuracy", color="red", marker="*", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2%}"))
    ax2.legend(loc="upper left")
    ax2.set_title("Accuracy (V2)")

    best_valid_acc_epoch = max(range(len(records)), key=lambda i: records[i].valid_acc) + 1
    best_valid_acc_record = records[best_valid_acc_epoch - 1]
    ax2.annotate(
        f"{best_valid_acc_record.valid_acc:.2%}",
        xy=(best_valid_acc_epoch, best_valid_acc_record.valid_acc),
        xytext=(best_valid_acc_epoch + 2, best_valid_acc_record.valid_acc),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def train_main(
    dataset_path: str,
    model_path: str,
    *,
    batch_size: int = 64,
    num_epochs: int = 100,
    patience_epochs: int = 15,
    lr: float = 1e-3,
    label_smoothing: float = 0.1,
    seed: int = 42,
    quiet: bool = False,
) -> tuple[list[TrainingRecord], float, float]:
    """Train ArkGuesserModelV2 and return (records, best_valid_acc, train_time_s)."""
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Dataset
    dataset = RawArkGuesserDataset(dataset_path)
    n_total = len(dataset)
    n_train = int(n_total * 0.9)
    n_valid = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_valid])
    train_set = AugArkGuesserDataset(dataset, subset=train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=AugArkGuesserDataset.collate_fn,
    )
    valid_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RawArkGuesserDataset.collate_fn,
    )
    if not quiet:
        print(f"Dataset loaded: {n_total} raw samples ({n_train} train, {n_valid} valid)")

    # Prepare training
    model = ArkGuesserModelV2(dataset.num_classes)
    model.to(device)
    if not quiet:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model V2 parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Run training
    worse_epochs = 0
    records = []
    best_epoch = 0
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validate_one_epoch(model, valid_loader, criterion, device)
        records.append(TrainingRecord(epoch, train_loss, train_acc, valid_loss, valid_acc))
        scheduler.step()

        if not quiet:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, acc={valid_acc:.2%}")

        if best_epoch == 0 or valid_loss * (1 - valid_acc) < (
            records[best_epoch - 1].valid_loss * (1 - records[best_epoch - 1].valid_acc)
        ):
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            worse_epochs = 0
        else:
            worse_epochs += 1
            if worse_epochs >= patience_epochs:
                if not quiet:
                    print(f"Early stopped at epoch {epoch}")
                break

    t1 = time.perf_counter()
    train_time = t1 - t0

    best_record = records[best_epoch - 1]
    if not quiet:
        print(f"Best epoch={best_epoch} with valid_loss={best_record.valid_loss:.4f}, acc={best_record.valid_acc:.2%}")

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(best_state, model_path)
    if not quiet:
        print(f"Best ckpt saved to {model_path}")

    visualize_records(records)

    return records, best_record.valid_acc, train_time
