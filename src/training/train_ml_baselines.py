import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "features"
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Train split-aware traditional ML baselines for BirdVoice.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", default="knn,svm,random_forest,xgboost")
    parser.add_argument("--svm-mode", choices=["approx", "exact"], default="approx")
    parser.add_argument("--svm-components", type=int, default=1024)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--comparison-out", default=str(OUT / "model_comparison.csv"))
    parser.add_argument("--cv-out", default=str(OUT / "logs" / "ml_baselines" / "cv_metrics.csv"))
    parser.add_argument("--confusion-out", default=str(OUT / "logs" / "ml_baselines" / "confusion_matrices.json"))
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    return parser.parse_args()


def load_feature_table():
    meta_path = FEATURES / "feature_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("缺少 features/feature_meta.json，请先运行 extract_features.py 提取特征")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = []
    labels = []
    splits = []
    indices = []
    for item in meta:
        path = FEATURES / "vectors" / f"{int(item['index']):07d}.npy"
        if path.exists():
            rows.append(np.load(path))
            labels.append(item["label"])
            splits.append(item.get("split", "all"))
            indices.append(int(item["index"]))
    if not rows:
        raise RuntimeError("未找到可训练特征向量")
    return np.vstack(rows), np.asarray(labels), np.asarray(splits), np.asarray(indices)


def build_svm_candidate(args):
    if args.svm_mode == "exact":
        return SVC(kernel="rbf", C=10, probability=True, class_weight="balanced")
    return make_pipeline(
        StandardScaler(),
        RBFSampler(gamma=0.05, n_components=args.svm_components, random_state=2026),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1200,
            tol=1e-3,
            class_weight="balanced",
            random_state=2026,
            n_jobs=-1,
        ),
    )


def build_candidates(args, selected=None):
    selected = set(selected or ["knn", "svm", "random_forest", "xgboost"])
    all_candidates = {
        "knn": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance"),
        "svm": build_svm_candidate(args),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_split=5,
            n_jobs=-1,
            random_state=2026,
        ),
    }
    candidates = {name: model for name, model in all_candidates.items() if name in selected}
    if not args.skip_xgboost and "xgboost" in selected:
        try:
            from xgboost import XGBClassifier

            candidates["xgboost"] = XGBClassifier(
                objective="multi:softprob",
                max_depth=8,
                learning_rate=0.1,
                n_estimators=args.xgb_estimators,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric=["mlogloss", "merror"],
                tree_method="hist",
                random_state=2026,
            )
        except Exception as exc:
            print(f"skip xgboost: {exc!r}")
    return candidates


def top5_score(model, x, y, class_labels):
    if not hasattr(model, "predict_proba"):
        return np.nan
    proba = model.predict_proba(x)
    return top_k_accuracy_score(y, proba, k=min(5, proba.shape[1]), labels=class_labels)


def evaluate_model(model, x_train, y_train, x_val, y_val, class_labels):
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    return {
        "accuracy": accuracy_score(y_val, pred),
        "macro_f1": f1_score(y_val, pred, average="macro"),
        "top5_acc": top5_score(model, x_val, y_val, class_labels),
        "prediction": pred,
    }


def cross_validate(name, model, x, y, folds, class_labels):
    _, counts = np.unique(y, return_counts=True)
    effective_folds = min(folds, int(counts.min())) if len(counts) else 0
    if effective_folds < 2:
        return []
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=2026)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y), start=1):
        metrics = evaluate_model(model, x[train_idx], y[train_idx], x[val_idx], y[val_idx], class_labels)
        rows.append(
            {
                "model": name,
                "fold": fold,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "top5_acc": metrics["top5_acc"],
            }
        )
    return rows


def main():
    args = parse_args()
    if args.check_config:
        print(
            json.dumps(
                {
                    "feature_meta": str(FEATURES / "feature_meta.json"),
                    "split_manifest": str(DATA / "split_manifest.csv"),
                    "model_out": str(MODELS),
                    "comparison_out": args.comparison_out,
                    "cv_out": args.cv_out,
                    "confusion_out": args.confusion_out,
                    "svm_mode": args.svm_mode,
                    "svm_components": args.svm_components,
                    "xgb_estimators": args.xgb_estimators,
                },
                ensure_ascii=False,
            )
        )
        return

    x, labels, splits, _ = load_feature_table()
    if {"train", "val"}.issubset(set(splits)):
        split_stats = pd.DataFrame({"label": labels, "split": splits}).groupby(["label", "split"]).size().unstack(fill_value=0)
        eligible_labels = split_stats[(split_stats.get("train", 0) >= 2) & (split_stats.get("val", 0) >= 1)].index.to_numpy()
        eligible_mask = np.isin(labels, eligible_labels) & np.isin(splits, ["train", "val"])
        x = x[eligible_mask]
        labels = labels[eligible_mask]
        splits = splits[eligible_mask]
        if len(np.unique(labels)) < 2:
            raise RuntimeError("可训练类别不足：需要至少 2 个同时包含 train/val 的类别")

    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    class_labels = np.arange(len(enc.classes_))
    MODELS.mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / "ml_baselines").mkdir(parents=True, exist_ok=True)

    if {"train", "val"}.issubset(set(splits)):
        train_mask = splits == "train"
        val_mask = splits == "val"
    else:
        train_mask = np.ones(len(y), dtype=bool)
        val_mask = np.ones(len(y), dtype=bool)

    selected_models = [name.strip() for name in args.models.split(",") if name.strip()]
    candidates = build_candidates(args, selected=selected_models)
    cv_rows = []
    final_rows = []
    confusion_payload = {}

    for name, model in candidates.items():
        cv_rows.extend(cross_validate(name, model, x[train_mask], y[train_mask], args.folds, class_labels))
        metrics = evaluate_model(model, x[train_mask], y[train_mask], x[val_mask], y[val_mask], class_labels)
        final_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "top5_acc": metrics["top5_acc"],
            }
        )
        confusion_payload[name] = confusion_matrix(y[val_mask], metrics["prediction"], labels=class_labels).tolist()
        joblib.dump({"model": model, "label_encoder": enc}, MODELS / f"{name}.joblib")

    cv_out = Path(args.cv_out)
    comparison_out = Path(args.comparison_out)
    confusion_out = Path(args.confusion_out)
    cv_out.parent.mkdir(parents=True, exist_ok=True)
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    confusion_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(cv_out, index=False)
    pd.DataFrame(final_rows).to_csv(comparison_out, index=False)
    confusion_out.write_text(
        json.dumps(confusion_payload),
        encoding="utf-8",
    )
    print(pd.DataFrame(final_rows).to_string(index=False))


if __name__ == "__main__":
    main()
