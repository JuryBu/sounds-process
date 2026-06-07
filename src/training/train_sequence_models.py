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
    parser = argparse.ArgumentParser(description="Train BirdVoice sequence/attention models.")
    parser.add_argument("--model", choices=["bilstm", "ast", "mamba", "ast_stub", "mamba_stub"], default="bilstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--smoke", action="store_true")
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


def require_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    return torch, nn, DataLoader


class SequenceDataset:
    def __init__(self, rows, label_index, torch):
        self.rows = rows
        self.label_index = label_index
        self.torch = torch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        vector_path = FEATURES / "vectors" / f"{int(row['index']):07d}.npy"
        vector = np.load(vector_path).astype("float32")
        sequence = np.tile(vector[:40], (64, 1))
        target = self.label_index[row["label"]]
        return self.torch.from_numpy(sequence), self.torch.tensor(target, dtype=self.torch.long)


class BiLstmClassifier:
    def __init__(self, class_count):
        torch, nn, _ = require_torch()
        self.module = nn.Sequential()
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.3), nn.Linear(512, class_count))

    def to(self, device):
        self.lstm = self.lstm.to(device)
        self.head = self.head.to(device)
        return self

    def train(self):
        self.lstm.train()
        self.head.train()

    def eval(self):
        self.lstm.eval()
        self.head.eval()

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.head.parameters())

    def state_dict(self):
        return {"lstm": self.lstm.state_dict(), "head": self.head.state_dict()}

    def __call__(self, sequence):
        output, _ = self.lstm(sequence)
        pooled = output.mean(dim=1)
        return self.head(pooled)


class AstLikeClassifier:
    def __init__(self, class_count):
        torch, nn, _ = require_torch()
        self.proj = nn.Linear(40, 128)
        self.pos = nn.Parameter(torch.zeros(1, 64, 128))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.25), nn.Linear(128, class_count))

    def to(self, device):
        _, nn, _ = require_torch()
        self.proj = self.proj.to(device)
        self.pos = nn.Parameter(self.pos.to(device))
        self.encoder = self.encoder.to(device)
        self.head = self.head.to(device)
        return self

    def train(self):
        self.proj.train()
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.proj.eval()
        self.encoder.eval()
        self.head.eval()

    def parameters(self):
        return list(self.proj.parameters()) + [self.pos] + list(self.encoder.parameters()) + list(self.head.parameters())

    def state_dict(self):
        return {
            "proj": self.proj.state_dict(),
            "pos": self.pos.detach().cpu(),
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
        }

    def __call__(self, sequence):
        x = self.proj(sequence) + self.pos[:, : sequence.shape[1], :]
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


class MambaLikeClassifier:
    def __init__(self, class_count):
        torch, nn, _ = require_torch()
        self.proj = nn.Linear(40, 128)
        self.conv = nn.Conv1d(128, 128, kernel_size=5, padding=2, groups=8)
        self.gate = nn.Sequential(nn.Linear(128, 128), nn.SiLU())
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.25), nn.Linear(128, class_count))

    def to(self, device):
        self.proj = self.proj.to(device)
        self.conv = self.conv.to(device)
        self.gate = self.gate.to(device)
        self.gru = self.gru.to(device)
        self.head = self.head.to(device)
        return self

    def train(self):
        self.proj.train()
        self.conv.train()
        self.gate.train()
        self.gru.train()
        self.head.train()

    def eval(self):
        self.proj.eval()
        self.conv.eval()
        self.gate.eval()
        self.gru.eval()
        self.head.eval()

    def parameters(self):
        return (
            list(self.proj.parameters())
            + list(self.conv.parameters())
            + list(self.gate.parameters())
            + list(self.gru.parameters())
            + list(self.head.parameters())
        )

    def state_dict(self):
        return {
            "proj": self.proj.state_dict(),
            "conv": self.conv.state_dict(),
            "gate": self.gate.state_dict(),
            "gru": self.gru.state_dict(),
            "head": self.head.state_dict(),
        }

    def __call__(self, sequence):
        x = self.proj(sequence)
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + conv * self.gate(x)
        output, _ = self.gru(x)
        return self.head(output.mean(dim=1))


def build_model(model_name, class_count):
    if model_name == "bilstm":
        return BiLstmClassifier(class_count)
    if model_name in {"ast", "ast_stub"}:
        return AstLikeClassifier(class_count)
    if model_name in {"mamba", "mamba_stub"}:
        return MambaLikeClassifier(class_count)
    raise NotImplementedError(model_name)


def evaluate(model, loader, criterion, device):
    torch, _, _ = require_torch()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    targets = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for sequence, target in loader:
            sequence = sequence.to(device)
            target = target.to(device)
            logits = model(sequence)
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

    torch, nn, DataLoader = require_torch()
    rows = load_feature_meta()
    train_rows, val_rows = split_rows(rows, args.limit)
    if len(train_rows) < 2 or len(val_rows) < 1:
        raise RuntimeError("特征样本不足，无法训练序列模型")
    label_index = build_label_index(train_rows + val_rows)
    train_dataset = SequenceDataset(train_rows, label_index, torch)
    val_dataset = SequenceDataset(val_rows, label_index, torch)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    model = build_model(args.model, len(label_index)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.smoke:
        sequence, _ = next(iter(train_loader))
        logits = model(sequence.to(device))
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
        for sequence, target in train_loader:
            sequence = sequence.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence)
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

    MODELS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "logs" / args.model).mkdir(parents=True, exist_ok=True)
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
