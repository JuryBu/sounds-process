import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate, ensemble, and summarize BirdVoice model predictions.")
    parser.add_argument("--prediction-dir", default=str(OUTPUTS / "predictions"))
    parser.add_argument("--labels", default=str(OUTPUTS / "labels.csv"))
    parser.add_argument("--out", default=str(OUTPUTS / "model_comparison_final.csv"))
    parser.add_argument("--check-config", action="store_true", help="Print expected inputs without requiring prediction files.")
    return parser.parse_args()


def top_k_accuracy(probabilities, labels, k=5):
    top_indices = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(labels, top_indices)]))


def evaluate_probabilities(probabilities, labels):
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "top5_accuracy": top_k_accuracy(probabilities, labels, k=5),
    }


def evaluate_prediction(predicted, labels, model_name, model_type):
    return {
        "model": model_name,
        "type": model_type,
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "top5_accuracy": np.nan,
    }


def load_predictions(prediction_dir):
    rows = []
    for path in sorted(Path(prediction_dir).glob("*.npy")):
        rows.append({"model": path.stem, "probabilities": np.load(path)})
    if not rows:
        raise FileNotFoundError(f"未找到预测概率文件：{prediction_dir}/*.npy")
    return rows


def weighted_average(prediction_rows, weights):
    total = None
    for row in prediction_rows:
        weight = weights.get(row["model"], 1.0)
        contribution = row["probabilities"] * weight
        total = contribution if total is None else total + contribution
    return total / sum(weights.values())


def majority_vote(prediction_rows):
    predictions = np.vstack([row["probabilities"].argmax(axis=1) for row in prediction_rows]).T
    class_count = prediction_rows[0]["probabilities"].shape[1]
    voted = []
    for row in predictions:
        voted.append(int(np.bincount(row, minlength=class_count).argmax()))
    return np.asarray(voted)


def stacking_holdout(prediction_rows, labels):
    features = np.hstack([row["probabilities"] for row in prediction_rows])
    counts = pd.Series(labels).value_counts()
    eligible_classes = counts[counts >= 2].index.to_numpy()
    mask = np.isin(labels, eligible_classes)
    x = features[mask]
    y = labels[mask]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.5,
        random_state=2026,
        stratify=y,
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1200,
        tol=1e-3,
        random_state=2026,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    return {
        "model": "stacking_holdout",
        "type": "ensemble",
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "top5_accuracy": top_k_accuracy(probabilities, y_test, k=5),
        "holdout_rows": int(len(y_test)),
        "classes": int(len(np.unique(y))),
    }


def main(args):
    if args.check_config:
        print(
            json.dumps(
                {
                    "prediction_dir": args.prediction_dir,
                    "labels": args.labels,
                    "out": args.out,
                },
                ensure_ascii=False,
            )
        )
        return

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError("缺少 labels.csv，应包含 sample_id,label,label_index")
    labels_df = pd.read_csv(labels_path)
    labels = labels_df["label_index"].to_numpy()
    prediction_rows = load_predictions(args.prediction_dir)

    metrics = []
    weights = {}
    for row in prediction_rows:
        metric = evaluate_probabilities(row["probabilities"], labels)
        weights[row["model"]] = max(metric["macro_f1"], 1e-6)
        metrics.append({"model": row["model"], "type": "single", **metric})

    ensemble_probabilities = weighted_average(prediction_rows, weights)
    metrics.append({"model": "weighted_average", "type": "ensemble", **evaluate_probabilities(ensemble_probabilities, labels)})
    metrics.append(evaluate_prediction(majority_vote(prediction_rows), labels, "majority_vote", "ensemble"))
    metrics.append(stacking_holdout(prediction_rows, labels))

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).sort_values("macro_f1", ascending=False).to_csv(output_path, index=False)

    predicted = ensemble_probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted).tolist()
    (OUTPUTS / "confusion_matrix_weighted_average.json").write_text(
        json.dumps(matrix),
        encoding="utf-8",
    )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main(parse_args())
