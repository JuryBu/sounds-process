import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FEATURES = ROOT / "features"
OUTPUTS = ROOT / "outputs"
SR = 32000
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
TARGET_SECONDS = 5
TARGET_LEN = SR * TARGET_SECONDS


def parse_args():
    parser = argparse.ArgumentParser(description="Extract split-aware BirdVoice features.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument("--validate-samples", type=int, default=10)
    parser.add_argument("--skip-pitch", action="store_true", help="Skip librosa.pyin when speed is more important.")
    return parser.parse_args()


def load_manifest():
    manifest_path = DATA / "split_manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)

    train_csv = DATA / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError("缺少 data/train.csv，请先完成 Kaggle 下载与解压")
    df = pd.read_csv(train_csv)
    label_col = "primary_label" if "primary_label" in df.columns else df.columns[0]
    manifest = df.copy()
    manifest.insert(0, "index", range(len(df)))
    manifest["split"] = "all"
    manifest["label"] = manifest[label_col].astype(str)
    manifest["audio_path"] = manifest.apply(audio_path_from_row, axis=1)
    return manifest


def audio_path_from_row(row):
    if "audio_path" in row and pd.notna(row["audio_path"]) and str(row["audio_path"]):
        audio_path = str(row["audio_path"]).replace("\\", "/")
        return audio_path if audio_path.startswith("train_audio/") else f"train_audio/{audio_path}"
    if "filename" in row and pd.notna(row["filename"]):
        filename = str(row["filename"]).replace("\\", "/")
        if filename.startswith("train_audio/"):
            return filename
        if "/" in filename:
            return f"train_audio/{filename}"
        label = str(row.get("primary_label", row.get("label", "")))
        return f"train_audio/{label}/{filename}"
    label = str(row.get("primary_label", row.get("label", "")))
    filename = str(row.get("filename", ""))
    return f"train_audio/{label}/{filename}"


def fix_length(y):
    if len(y) >= TARGET_LEN:
        return y[:TARGET_LEN]
    return np.pad(y, (0, TARGET_LEN - len(y)))


def feature_vector(y, skip_pitch=False):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    energy = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)

    blocks = [mfcc, delta, energy, centroid, bandwidth, rolloff, zcr]
    values = []
    for block in blocks:
        values.extend(np.mean(block, axis=1))
        values.extend(np.std(block, axis=1))

    values.extend(pitch_features(y, skip_pitch=skip_pitch))
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def pitch_features(y, skip_pitch=False):
    if skip_pitch:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SR,
            hop_length=HOP_LENGTH,
        )
        voiced = f0[~np.isnan(f0)]
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
        if len(voiced) == 0:
            return [0.0, 0.0, 0.0, voiced_ratio]
        return [float(np.mean(voiced)), float(np.std(voiced)), float(np.median(voiced)), voiced_ratio]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def mel_image(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def validate_features(meta, sample_count):
    rows = []
    for item in meta[:sample_count]:
        vector_path = FEATURES / "vectors" / f"{item['index']:07d}.npy"
        mel_path = FEATURES / "mel" / f"{item['index']:07d}.npy"
        vector = np.load(vector_path)
        mel = np.load(mel_path)
        rows.append(
            {
                "index": int(item["index"]),
                "label": item["label"],
                "split": item.get("split", "all"),
                "vector_shape": list(vector.shape),
                "mel_shape": list(mel.shape),
                "vector_min": float(np.min(vector)),
                "vector_max": float(np.max(vector)),
                "vector_has_nan": bool(np.isnan(vector).any()),
                "mel_has_nan": bool(np.isnan(mel).any()),
            }
        )
    out = OUTPUTS / "logs" / "feature_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    args = parse_args()
    df = load_manifest()
    if args.per_class_limit:
        sort_cols = [col for col in ["label", "split", "index"] if col in df.columns]
        df = (
            df.sort_values(sort_cols)
            .groupby(["label", "split"] if "split" in df.columns else ["label"], group_keys=False)
            .head(args.per_class_limit)
            .sort_values("index")
        )
    if args.limit:
        df = df.head(args.limit)

    (FEATURES / "vectors").mkdir(parents=True, exist_ok=True)
    (FEATURES / "mel").mkdir(parents=True, exist_ok=True)
    meta = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="features"):
        path = DATA / audio_path_from_row(row)
        if not path.exists():
            continue
        y, _ = librosa.load(path, sr=SR, mono=True, duration=TARGET_SECONDS)
        y = fix_length(y)

        label = str(row.get("label", row.get("primary_label", "")))
        split = str(row.get("split", "all"))
        index = int(row["index"])
        stem = f"{index:07d}"
        np.save(FEATURES / "vectors" / f"{stem}.npy", feature_vector(y, skip_pitch=args.skip_pitch))
        np.save(FEATURES / "mel" / f"{stem}.npy", mel_image(y))
        meta.append(
            {
                "index": index,
                "label": label,
                "label_index": int(row["label_index"]) if "label_index" in row and pd.notna(row["label_index"]) else None,
                "split": split,
                "source": str(path.relative_to(DATA)),
            }
        )

    (FEATURES / "feature_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(meta).to_csv(FEATURES / "feature_meta.csv", index=False)
    if meta:
        labels = pd.DataFrame(meta)[["index", "label", "label_index", "split", "source"]]
        (OUTPUTS).mkdir(parents=True, exist_ok=True)
        labels.to_csv(OUTPUTS / "labels.csv", index=False)
        validation = validate_features(meta, min(args.validate_samples, len(meta)))
    else:
        validation = []
    print(json.dumps({"features": len(meta), "out": str(FEATURES), "validation_samples": len(validation)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
