"""
cv/presence/train_presence.py

Trains Model A: MobileNetV3-Small Binary Presence Classifier.
Label Mapping:
  0: NOT_NOTE (Empty floor, paper, receipts, coins, wrappers, ambient shadows)
  1: NOTE     (Indian Banknotes: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500)

Saves:
  - Checkpoint: cv/presence/presence_classifier_v1.pt
  - Config: cv/presence/presence_config.json
  - Class Mapping: cv/presence/class_mapping.json
  - Training History: cv/presence/training_history.json
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESENCE_DIR = PROJECT_ROOT / "presence"
DATASET_DIR = PRESENCE_DIR / "dataset"

DEVICE = torch.device("cpu")  # Train cleanly on CPU (or CUDA if available)
BATCH_SIZE = 32
NUM_EPOCHS = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
INPUT_SIZE = (224, 224)
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CLASS_TO_IDX = {"NOT_NOTE": 0, "NOTE": 1}
IDX_TO_CLASS = {0: "NOT_NOTE", 1: "NOTE"}


class ExplicitBinaryDataset(datasets.ImageFolder):
    def find_classes(self, dir: str):
        classes = ["NOT_NOTE", "NOTE"]
        class_to_idx = CLASS_TO_IDX
        return classes, class_to_idx


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ExplicitBinaryDataset(str(DATASET_DIR / "train"), transform=train_transform)
    val_dataset = ExplicitBinaryDataset(str(DATASET_DIR / "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Dataset classes explicitly mapped: {train_dataset.class_to_idx}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def build_model(num_classes=2):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def compute_metrics(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    acc = (tp + tn) / max(1, len(y_true))
    # NOTE metrics (class 1)
    prec_note = tp / max(1, (tp + fp))
    rec_note = tp / max(1, (tp + fn))
    f1_note = 2 * prec_note * rec_note / max(1e-8, (prec_note + rec_note))

    # NOT_NOTE metrics (class 0)
    prec_not_note = tn / max(1, (tn + fn))
    rec_not_note = tn / max(1, (tn + fp))
    f1_not_note = 2 * prec_not_note * rec_not_note / max(1e-8, (prec_not_note + rec_not_note))

    fpr = fp / max(1, (fp + tn))
    return {
        "accuracy": round(acc, 4),
        "note": {
            "precision": round(prec_note, 4),
            "recall": round(rec_note, 4),
            "f1": round(f1_note, 4),
        },
        "not_note": {
            "precision": round(prec_not_note, 4),
            "recall": round(rec_not_note, 4),
            "f1": round(f1_not_note, 4),
        },
        "fpr": round(fpr, 4),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        }
    }


def train():
    print("=" * 60)
    print("Training Model A: MobileNetV3-Small Binary Presence Classifier")
    print(f"Device: {DEVICE}")
    print(f"Label Mapping: {CLASS_TO_IDX}")
    print("=" * 60)

    train_loader, val_loader = get_dataloaders()
    model = build_model(num_classes=2)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_f1 = -1.0
    best_epoch = 0
    best_weights = None
    history = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss /= total_train
        train_acc = train_correct / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Val]  "):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        val_loss /= len(y_true)
        val_metrics = compute_metrics(y_true, y_pred)
        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_note_precision": val_metrics["note"]["precision"],
            "val_note_recall": val_metrics["note"]["recall"],
            "val_note_f1": val_metrics["note"]["f1"],
            "val_not_note_f1": val_metrics["not_note"]["f1"],
            "val_fpr": val_metrics["fpr"],
            "epoch_duration_sec": round(epoch_time, 2),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, "
            f"NOTE F1: {val_metrics['note']['f1']:.4f}, NOT_NOTE F1: {val_metrics['not_note']['f1']:.4f}, FPR: {val_metrics['fpr']*100:.2f}%"
        )

        if val_metrics["note"]["f1"] > best_val_f1:
            best_val_f1 = val_metrics["note"]["f1"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  --> Saved new best checkpoint at Epoch {epoch} (Val NOTE F1: {best_val_f1:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best epoch: {best_epoch}")

    # Save model and artifacts
    ckpt_path = PRESENCE_DIR / "presence_classifier_v1.pt"
    torch.save(best_weights, ckpt_path)
    print(f"Best model weights saved to: {ckpt_path}")

    # Class mapping
    mapping_path = PRESENCE_DIR / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({"class_to_idx": CLASS_TO_IDX, "idx_to_class": IDX_TO_CLASS}, f, indent=2)

    # Config
    config = {
        "model_name": "MobileNetV3-Small Binary Presence Classifier",
        "checkpoint_file": "presence_classifier_v1.pt",
        "checkpoint_path": str(ckpt_path),
        "random_seed": SEED,
        "input_size": list(INPUT_SIZE),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "hyperparameters": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR"
        },
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "total_training_time_sec": round(total_time, 2)
    }
    config_path = PRESENCE_DIR / "presence_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # History
    history_path = PRESENCE_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Config saved to: {config_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    train()
