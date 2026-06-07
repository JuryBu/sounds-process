import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "features"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune official MIT AST on BirdVoice Mel features.")
    parser.add_argument("--pretrained", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def load_rows(limit):
    meta_path = FEATURES / "feature_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("缺少 features/feature_meta.json")
    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    return rows[:limit] if limit else rows


def label_index(rows):
    labels = sorted({row["label"] for row in rows})
    return {label: index for index, label in enumerate(labels)}


def ast_input(row, target_frames=1024):
    mel = np.load(FEATURES / "mel" / f"{int(row['index']):07d}.npy").astype("float32")
    matrix = mel.T
    if matrix.shape[0] >= target_frames:
        matrix = matrix[:target_frames]
    else:
        pad = np.zeros((target_frames - matrix.shape[0], matrix.shape[1]), dtype="float32")
        matrix = np.vstack([matrix, pad])
    mean = float(matrix.mean())
    std = float(matrix.std() + 1e-6)
    return (matrix - mean) / std


class AstDataset:
    def __init__(self, rows, labels, torch):
        self.rows = rows
        self.labels = labels
        self.torch = torch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        inputs = ast_input(row)
        target = self.labels[row["label"]]
        return self.torch.from_numpy(inputs), self.torch.tensor(target, dtype=self.torch.long)


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader
    from transformers import ASTForAudioClassification

    rows = load_rows(args.limit)
    labels = label_index(rows)
    dataset = AstDataset(rows, labels, torch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    model = ASTForAudioClassification.from_pretrained(
        args.pretrained,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for input_values, target in loader:
            input_values = input_values.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_values=input_values, labels=target)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * target.numel()
            total += target.numel()
            correct += (output.logits.argmax(dim=1) == target).sum().item()
        row = {
            "epoch": epoch,
            "loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "pretrained": args.pretrained,
            "samples": len(dataset),
            "classes": len(labels),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    MODELS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "logs" / "official_ast").mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "label_index": labels, "pretrained": args.pretrained}, MODELS / "ast_official.pth")
    (OUTPUTS / "logs" / "official_ast" / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
