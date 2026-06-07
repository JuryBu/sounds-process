import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "eda"


def parse_args():
    parser = argparse.ArgumentParser(description="Run BirdCLEF2026 EDA and save reproducible artifacts.")
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--plot-examples", type=int, default=3)
    return parser.parse_args()


def find_audio_files(audio_dir):
    return sorted(audio_dir.rglob("*.ogg")) + sorted(audio_dir.rglob("*.wav")) + sorted(audio_dir.rglob("*.mp3"))


def plot_wave_and_mel(path, index):
    y, sr = librosa.load(path, sr=None, mono=True, duration=30)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform: {path.relative_to(DATA)}")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1])
    axes[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.savefig(OUT / f"sample_{index:02d}_wave_mel.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    taxonomy = DATA / "taxonomy.csv"
    train_csv = DATA / "train.csv"
    train_audio = DATA / "train_audio"

    if not taxonomy.exists() or not train_csv.exists() or not train_audio.exists():
        raise FileNotFoundError("缺少 taxonomy.csv / train.csv / train_audio，请先完成 Kaggle 下载与解压")

    OUT.mkdir(parents=True, exist_ok=True)
    tax = pd.read_csv(taxonomy)
    train = pd.read_csv(train_csv)
    audio_files = find_audio_files(train_audio)

    label_col = "primary_label" if "primary_label" in train.columns else train.columns[0]
    counts = train[label_col].astype(str).value_counts()
    imbalance = counts.rename_axis("label").reset_index(name="count")
    imbalance["ratio_to_max"] = imbalance["count"] / imbalance["count"].max()
    imbalance.to_csv(OUT / "class_imbalance.csv", index=False)

    sample_stats = []
    for path in audio_files[: args.sample_count]:
        try:
            y, sr = librosa.load(path, sr=None, mono=True, duration=30)
            sample_stats.append(
                {
                    "file": str(path.relative_to(DATA)),
                    "sample_rate": int(sr),
                    "duration": float(len(y) / sr),
                    "samples": int(len(y)),
                }
            )
        except Exception as exc:
            sample_stats.append({"file": str(path.relative_to(DATA)), "error": repr(exc)})

    ok_stats = [row for row in sample_stats if "duration" in row]
    sample_rate_counts = (
        pd.DataFrame(ok_stats)["sample_rate"].value_counts().sort_index().to_dict() if ok_stats else {}
    )
    summary = {
        "taxonomy_rows": int(len(tax)),
        "train_rows": int(len(train)),
        "audio_files": int(len(audio_files)),
        "columns_taxonomy": list(tax.columns),
        "columns_train": list(train.columns),
        "class_count": int(counts.shape[0]),
        "min_samples_per_class": int(counts.min()),
        "max_samples_per_class": int(counts.max()),
        "sample_rate_counts": {str(key): int(value) for key, value in sample_rate_counts.items()},
        "sample_audio_stats": sample_stats,
        "imbalance_top10": imbalance.head(10).to_dict(orient="records"),
        "imbalance_bottom10": imbalance.tail(10).to_dict(orient="records"),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 5))
    sns.histplot(counts.values, bins=50)
    plt.title("Samples per class")
    plt.xlabel("sample count")
    plt.ylabel("class count")
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=180)
    plt.close()

    if ok_stats:
        df = pd.DataFrame(ok_stats)
        df.to_csv(OUT / "sample_audio_stats.csv", index=False)
        plt.figure(figsize=(12, 5))
        sns.histplot(df["duration"], bins=30)
        plt.title(f"Sample audio duration (first {len(df)} files)")
        plt.xlabel("seconds")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(OUT / "duration_sample.png", dpi=180)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x="sample_rate")
        plt.title("Sample rate check")
        plt.xlabel("sample rate")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(OUT / "sample_rate_distribution.png", dpi=180)
        plt.close()

    for index, path in enumerate(audio_files[: args.plot_examples], start=1):
        try:
            plot_wave_and_mel(path, index)
        except Exception as exc:
            (OUT / f"sample_{index:02d}_plot_error.txt").write_text(repr(exc), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
