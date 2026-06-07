import argparse
import json
from pathlib import Path

import librosa
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "logs" / "integrity_report.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate BirdCLEF2026 dataset layout and sample readability.")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--json-out", default=str(ROOT / "outputs" / "logs" / "integrity_report.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    required = ["taxonomy.csv", "train.csv", "train_audio", "train_soundscapes"]
    lines = ["# BirdVoice 数据完整性报告", ""]
    payload = {"required": {}, "samples": []}
    ok = True

    for name in required:
        path = DATA / name
        exists = path.exists()
        payload["required"][name] = exists
        lines.append(f"- `{name}`: {'OK' if exists else 'MISSING'}")
        ok = ok and exists

    if (DATA / "taxonomy.csv").exists():
        tax = pd.read_csv(DATA / "taxonomy.csv")
        payload["taxonomy_rows"] = int(len(tax))
        payload["taxonomy_columns"] = list(tax.columns)
        lines.append(f"- taxonomy rows: {len(tax)}")
        lines.append(f"- taxonomy columns: {', '.join(tax.columns)}")

    if (DATA / "train.csv").exists():
        train = pd.read_csv(DATA / "train.csv")
        label_col = "primary_label" if "primary_label" in train.columns else train.columns[0]
        payload["train_rows"] = int(len(train))
        payload["class_count"] = int(train[label_col].nunique())
        payload["label_col"] = label_col
        lines.append(f"- train rows: {len(train)}")
        lines.append(f"- class count: {train[label_col].nunique()}")

    audio_files = sorted((DATA / "train_audio").rglob("*.ogg")) if (DATA / "train_audio").exists() else []
    soundscape_files = (
        sorted((DATA / "train_soundscapes").rglob("*.ogg"))
        + sorted((DATA / "train_soundscapes").rglob("*.wav"))
        if (DATA / "train_soundscapes").exists()
        else []
    )
    payload["train_audio_files"] = len(audio_files)
    payload["train_soundscape_files"] = len(soundscape_files)
    lines.append(f"- train audio files: {len(audio_files)}")
    lines.append(f"- train soundscape files: {len(soundscape_files)}")

    for sample in audio_files[: args.sample_count]:
        try:
            y, sr = librosa.load(sample, sr=None, mono=True, duration=10)
            row = {"file": str(sample.relative_to(DATA)), "sample_rate": int(sr), "duration": len(y) / sr}
            payload["samples"].append(row)
            lines.append(f"  - sample `{row['file']}` sr={sr} duration≈{row['duration']:.2f}s")
        except Exception as exc:
            row = {"file": str(sample.relative_to(DATA)), "error": repr(exc)}
            payload["samples"].append(row)
            lines.append(f"  - sample `{row['file']}` ERROR {exc!r}")
            ok = False

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
