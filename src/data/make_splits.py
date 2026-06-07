import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Create frozen stratified train/val/test splits.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def resolve_audio_path(row):
    if "filename" in row and pd.notna(row["filename"]):
        filename = str(row["filename"])
        if "/" in filename or "\\" in filename:
            filename = filename.replace("\\", "/")
            return filename if filename.startswith("train_audio/") else f"train_audio/{filename}"
    label = str(row.get("primary_label", ""))
    filename = str(row.get("filename", row.get("secondary_labels", "")))
    if filename and filename != "nan":
        return f"train_audio/{label}/{filename}".replace("\\", "/")
    return ""


def main():
    args = parse_args()
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")

    train_csv = DATA / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError("缺少 train.csv，请先完成 Kaggle 下载与解压")

    df = pd.read_csv(train_csv)
    label_col = "primary_label" if "primary_label" in df.columns else df.columns[0]
    labels = df[label_col].astype(str)
    train_idx, val_idx, test_idx = split_by_label(labels, args)

    label_map = {label: index for index, label in enumerate(sorted(labels.unique()))}
    split_by_index = {}
    for split, split_indices in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        for index in split_indices:
            split_by_index[index] = split

    manifest = df.copy()
    manifest.insert(0, "index", range(len(df)))
    manifest["split"] = manifest["index"].map(split_by_index)
    manifest["label"] = labels
    manifest["label_index"] = manifest["label"].map(label_map)
    manifest["audio_path"] = manifest.apply(resolve_audio_path, axis=1)
    manifest = manifest[["index", "split", "label", "label_index", "audio_path"] + [c for c in df.columns if c not in {"index"}]]

    payload = {
        "label_col": label_col,
        "seed": args.seed,
        "label_map_path": "data/label_map.json",
        "manifest_path": "data/split_manifest.csv",
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    }
    stats = (
        manifest.groupby(["split", "label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "label"])
    )

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "splits.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.to_csv(DATA / "split_manifest.csv", index=False)
    stats.to_csv(DATA / "split_stats.csv", index=False)
    print(json.dumps(payload["counts"], ensure_ascii=False))

def split_by_label(labels, args):
    import numpy as np

    rng = np.random.default_rng(args.seed)
    train_idx = []
    val_idx = []
    test_idx = []
    labels = pd.Series(labels).reset_index(drop=True)

    for _, group in labels.groupby(labels):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        total = len(indices)
        if total >= 3:
            val_count = max(1, int(round(total * args.val_ratio)))
            test_count = max(1, int(round(total * args.test_ratio)))
            if val_count + test_count >= total:
                val_count = 1
                test_count = 1
        elif total == 2:
            val_count = 1
            test_count = 0
        else:
            val_count = 0
            test_count = 0

        val_idx.extend(indices[:val_count].tolist())
        test_idx.extend(indices[val_count : val_count + test_count].tolist())
        train_idx.extend(indices[val_count + test_count :].tolist())

    for bucket in (train_idx, val_idx, test_idx):
        rng.shuffle(bucket)
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    main()
