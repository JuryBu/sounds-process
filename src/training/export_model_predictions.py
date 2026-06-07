import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "features"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Export validation probabilities for trained sklearn models.")
    parser.add_argument("--models", default="knn,random_forest,svm,xgboost")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", default=str(OUTPUTS / "predictions"))
    parser.add_argument("--labels-out", default=str(OUTPUTS / "predictions" / "labels.csv"))
    return parser.parse_args()


def load_rows():
    meta_path = FEATURES / "feature_meta.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)
    json_path = FEATURES / "feature_meta.json"
    if not json_path.exists():
        raise FileNotFoundError("缺少 features/feature_meta.csv 或 feature_meta.json")
    return pd.read_json(json_path)


def load_vectors(rows):
    vectors = []
    for index in rows["index"].astype(int):
        vectors.append(np.load(FEATURES / "vectors" / f"{index:07d}.npy"))
    return np.vstack(vectors)


def aligned_probabilities(model, x, class_count):
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float32)
    classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
    aligned = np.zeros((probabilities.shape[0], class_count), dtype=np.float32)
    for column, class_index in enumerate(classes):
        aligned[:, int(class_index)] = probabilities[:, column]
    return aligned


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    rows = load_rows()
    payloads = {}
    class_sets = []
    for name in model_names:
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = joblib.load(path)
        encoder = payload["label_encoder"]
        payloads[name] = payload
        class_sets.append(set(map(str, encoder.classes_)))

    common_labels = set.intersection(*class_sets)
    selected = rows[(rows["split"] == args.split) & (rows["label"].astype(str).isin(common_labels))].copy()
    if selected.empty:
        raise RuntimeError(f"未找到 split={args.split} 且所有模型共有类别的样本")

    reference_encoder = next(iter(payloads.values()))["label_encoder"]
    selected["label_index"] = reference_encoder.transform(selected["label"].astype(str))
    x = load_vectors(selected)

    labels_out = Path(args.labels_out)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    selected[["index", "label", "label_index", "split", "source"]].to_csv(labels_out, index=False)

    class_count = len(reference_encoder.classes_)
    for name, payload in payloads.items():
        probabilities = aligned_probabilities(payload["model"], x, class_count)
        np.save(out_dir / f"{name}.npy", probabilities)
        print(f"{name}: {probabilities.shape}")
    print(f"labels={labels_out} rows={len(selected)} classes={class_count}")


if __name__ == "__main__":
    main()
