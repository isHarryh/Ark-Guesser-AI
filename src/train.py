import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from matplotlib import ticker, axes

from src.dataset import RawArkGuesserDataset, AugArkGuesserDataset
from src.model import ArkGuesserModelV1, ArkGuesserModelV2

TrainingRecord = namedtuple("TrainingRecord", ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
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


def visualize_records(records: list[TrainingRecord], output_path: str = "outputs/training_metrics.png"):
    if not records:
        return

    epochs = [r.epoch for r in records]
    train_losses = [r.train_loss for r in records]
    valid_losses = [r.valid_loss for r in records]
    train_accs = [r.train_acc for r in records]
    valid_accs = [r.valid_acc for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    assert isinstance(ax1, axes.Axes) and isinstance(ax2, axes.Axes)

    # Left subplot for losses
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: round(x)))
    ax1.plot(epochs, train_losses, label="Train Loss", color="blue", marker="x")
    ax1.plot(epochs, valid_losses, label="Valid Loss", color="red", marker="*")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.4f}"))
    ax1.legend(loc="upper left")
    ax1.set_title("Loss")

    best_valid_loss_epoch = min(range(len(records)), key=lambda i: records[i].valid_loss) + 1
    best_valid_loss_record = records[best_valid_loss_epoch - 1]
    ax1.annotate(
        f"{best_valid_loss_record.valid_loss:.4f}",
        xy=(best_valid_loss_epoch, best_valid_loss_record.valid_loss),
        xytext=(best_valid_loss_epoch + 2, best_valid_loss_record.valid_loss),
        arrowprops=dict(arrowstyle="->"),
    )

    # Right subplot for accuracies
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: round(x)))
    ax2.plot(epochs, train_accs, label="Train Accuracy", color="blue", marker="x", linestyle="--")
    ax2.plot(epochs, valid_accs, label="Valid Accuracy", color="red", marker="*", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2%}"))
    ax2.legend(loc="upper left")
    ax2.set_title("Accuracy")

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


def main(dataset_path: str, model_path: str, model_version: str = "v1"):
    # Config
    batch_size = 64
    num_epochs = 100
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
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
    print(f"Dataset loaded: {n_total} raw samples ({n_train} train, {n_valid} valid)")

    # Prepare model
    if model_version == "v2":
        model = ArkGuesserModelV2(dataset.num_classes)
        patience_epochs = 15
        print(f"Using model: ArkGuesserModelV2")
    else:
        model = ArkGuesserModelV1(dataset.num_classes)
        patience_epochs = max(2, num_epochs // 2)
        print(f"Using model: ArkGuesserModelV1")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Run training
    worse_epochs = 0
    records = []
    best_epoch = 0
    best_state = None
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validate_one_epoch(model, valid_loader, criterion, device)
        records.append(TrainingRecord(epoch, train_loss, train_acc, valid_loss, valid_acc))

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, acc={valid_acc:.2%}")

        if best_epoch == 0 or valid_loss * (1 - valid_acc) < (
            records[best_epoch - 1].valid_loss * (1 - records[best_epoch - 1].valid_acc)
        ):
            best_epoch = epoch
            best_state = model.state_dict()
            worse_epochs = 0
        else:
            worse_epochs += 1
            if worse_epochs >= patience_epochs:
                print("Early stopped")
                break

    best_record = records[best_epoch - 1]
    print(f"Best epoch={best_epoch} with valid_loss={best_record.valid_loss:.4f}, acc={best_record.valid_acc:.2%}")

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(best_state, model_path)
    print(f"Best ckpt saved to {model_path}")

    visualize_records(records)
