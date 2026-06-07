import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "features"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train BirdVoice CNN-family models on Mel spectrogram features.")
    parser.add_argument("--model", choices=["custom_cnn", "resnet50", "efficientnet_b2"], default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--smoke", action="store_true", help="Only validate dataset/model wiring and run a tiny pass.")
    parser.add_argument("--check-config", action="store_true", help="Print expected paths without requiring downloaded data.")
    return parser.parse_args()


def load_feature_meta():
    meta_path = FEATURES / "feature_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("缺少 features/feature_meta.json，请先运行 extract_features.py 提取特征")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def split_rows(rows, limit=None):
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    if train_rows and val_rows:
        if limit:
            train_limit = max(1, int(limit * 0.8))
            val_limit = max(1, limit - train_limit)
            train_rows = train_rows[:train_limit]
            val_rows = val_rows[:val_limit]
        return train_rows, val_rows

    selected = rows[:limit] if limit else rows
    rng = np.random.default_rng(2026)
    indices = np.arange(len(selected))
    rng.shuffle(indices)
    cut = max(1, int(len(indices) * 0.8))
    train_index = set(indices[:cut])
    train_rows = [row for index, row in enumerate(selected) if index in train_index]
    val_rows = [row for index, row in enumerate(selected) if index not in train_index]
    return train_rows, val_rows or train_rows[:1]


def build_label_index(rows):
    labels = sorted({row["label"] for row in rows})
    return {label: index for index, label in enumerate(labels)}


def load_mel_sample(row):
    mel_path = FEATURES / "mel" / f"{int(row['index']):07d}.npy"
    mel = np.load(mel_path).astype("float32")
    return mel[None, :, :]


def require_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    return torch, nn, DataLoader, Dataset


class MelDataset:
    def __init__(self, rows, label_index, dataset_base):
        self.rows = rows
        self.label_index = label_index
        self.dataset_base = dataset_base

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        torch = self.dataset_base["torch"]
        row = self.rows[index]
        mel = load_mel_sample(row)
        label = self.label_index[row["label"]]
        return torch.from_numpy(mel), torch.tensor(label, dtype=torch.long)


def build_custom_cnn(class_count):
    torch, nn, _, _ = require_torch()
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, class_count),
    )


def build_timm_model(model_name, class_count):
    import timm

    timm_name = {"resnet50": "resnet50", "efficientnet_b2": "efficientnet_b2"}[model_name]
    return timm.create_model(timm_name, pretrained=False, num_classes=class_count, in_chans=1)


def build_model(model_name, class_count):
    if model_name == "custom_cnn":
        return build_custom_cnn(class_count)
    return build_timm_model(model_name, class_count)


def evaluate(model, loader, criterion, device):
    torch, _, _, _ = require_torch()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    targets = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for mel, target in loader:
            mel = mel.to(device)
            target = target.to(device)
            logits = model(mel)
            loss = criterion(logits, target)
            pred = logits.argmax(dim=1)
            topk = min(5, logits.shape[1])
            top5 = logits.topk(topk, dim=1).indices
            total_loss += float(loss.item()) * target.numel()
            correct += (pred == target).sum().item()
            top5_correct += (top5 == target.unsqueeze(1)).any(dim=1).sum().item()
            total += target.numel()
            targets.extend(target.detach().cpu().numpy().tolist())
            predictions.extend(pred.detach().cpu().numpy().tolist())
    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)) if targets else 0.0,
        "top5_accuracy": top5_correct / total if total else 0.0,
    }


def train(args):
    if args.check_config:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "features_dir": str(FEATURES),
                    "feature_meta": str(FEATURES / "feature_meta.json"),
                    "model_out": str(MODELS / f"{args.model}.pth"),
                },
                ensure_ascii=False,
            )
        )
        return

    torch, nn, DataLoader, Dataset = require_torch()
    rows = load_feature_meta()
    train_rows, val_rows = split_rows(rows, args.limit)
    if len(train_rows) < 2 or len(val_rows) < 1:
        raise RuntimeError("特征样本不足，无法训练 CNN")

    label_index = build_label_index(train_rows + val_rows)
    train_dataset = MelDataset(train_rows, label_index, {"torch": torch})
    val_dataset = MelDataset(val_rows, label_index, {"torch": torch})
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    model = build_model(args.model, len(label_index)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    MODELS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "logs" / args.model).mkdir(parents=True, exist_ok=True)

    if args.smoke:
        mel, target = next(iter(train_loader))
        logits = model(mel.to(device))
        print(
            json.dumps(
                {
                    "model": args.model,
                    "logits_shape": list(logits.shape),
                    "classes": len(label_index),
                    "train_rows": len(train_rows),
                    "val_rows": len(val_rows),
                }
            )
        )
        return

    history = []
    best_metric = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for mel, target in train_loader:
            mel = mel.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mel)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == target).sum().item()
            total += target.numel()
            total_loss += float(loss.item()) * target.numel()
        val = evaluate(model, val_loader, criterion, device)
        train_loss = total_loss / len(train_dataset)
        train_accuracy = correct / total if total else 0.0
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_top5_accuracy": val["top5_accuracy"],
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val["accuracy"] > best_metric:
            best_metric = val["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    payload = {
        "model": best_state or model.state_dict(),
        "label_index": label_index,
        "best_val_accuracy": best_metric,
        "history": history,
    }
    torch.save(payload, MODELS / f"{args.model}.pth")
    torch.save(payload, MODELS / f"{args.model}_best.pth")
    (OUTPUTS / "logs" / args.model / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    train(parse_args())
