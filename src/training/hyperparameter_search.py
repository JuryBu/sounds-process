import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "features"
OUTPUTS = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight 5-fold hyperparameter search on real BirdCLEF features.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=30)
    parser.add_argument("--out", default=str(OUTPUTS / "logs" / "hyperparameter_search" / "knn_5fold_grid.csv"))
    parser.add_argument("--summary-out", default=str(OUTPUTS / "logs" / "hyperparameter_search" / "summary.json"))
    return parser.parse_args()


def load_meta():
    csv_path = FEATURES / "feature_meta.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    json_path = FEATURES / "feature_meta.json"
    if not json_path.exists():
        raise FileNotFoundError("缺少 features/feature_meta.csv 或 feature_meta.json")
    return pd.read_json(json_path)


def load_vectors(rows):
    vectors = []
    for index in rows["index"].astype(int):
        vectors.append(np.load(FEATURES / "vectors" / f"{index:07d}.npy"))
    return np.vstack(vectors)


def balanced_subset(meta, folds, max_per_class):
    train = meta[meta["split"] == "train"].copy()
    counts = train["label"].value_counts()
    eligible = counts[counts >= folds].index
    rows = []
    for label in eligible:
        label_rows = train[train["label"] == label].sort_values("index").head(max_per_class)
        rows.append(label_rows)
    if not rows:
        raise RuntimeError("没有满足 5-fold 的类别")
    return pd.concat(rows, ignore_index=True)


def main():
    args = parse_args()
    rows = balanced_subset(load_meta(), args.folds, args.max_per_class)
    x = load_vectors(rows)
    encoder = LabelEncoder()
    y = encoder.fit_transform(rows["label"].astype(str))
    labels = np.arange(len(encoder.classes_))
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    candidates = [
        {"k": 3, "metric": "cosine"},
        {"k": 5, "metric": "cosine"},
        {"k": 7, "metric": "cosine"},
        {"k": 5, "metric": "euclidean"},
    ]

    records = []
    for candidate in candidates:
        for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y), start=1):
            model = KNeighborsClassifier(
                n_neighbors=candidate["k"],
                metric=candidate["metric"],
                weights="distance",
            )
            model.fit(x[train_idx], y[train_idx])
            prediction = model.predict(x[val_idx])
            proba = model.predict_proba(x[val_idx])
            records.append(
                {
                    "model": "knn",
                    "k": candidate["k"],
                    "metric": candidate["metric"],
                    "fold": fold,
                    "accuracy": accuracy_score(y[val_idx], prediction),
                    "macro_f1": f1_score(y[val_idx], prediction, average="macro"),
                    "top5_acc": top_k_accuracy_score(y[val_idx], proba, k=min(5, proba.shape[1]), labels=labels),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(out, index=False)
    summary = (
        result.groupby(["model", "k", "metric"])[["accuracy", "macro_f1", "top5_acc"]]
        .mean()
        .reset_index()
        .sort_values(["macro_f1", "accuracy"], ascending=False)
    )
    summary_payload = {
        "rows": int(len(rows)),
        "classes": int(len(encoder.classes_)),
        "folds": args.folds,
        "max_per_class": args.max_per_class,
        "best": summary.iloc[0].to_dict(),
        "summary": summary.to_dict(orient="records"),
    }
    Path(args.summary_out).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
