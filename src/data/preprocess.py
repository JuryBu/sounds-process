from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "data" / "processed_audio"
TARGET_SR = 32000
TARGET_SECONDS = 5
TARGET_LEN = TARGET_SR * TARGET_SECONDS
RNG = np.random.default_rng(2026)


def fix_length(y, train=True):
    if len(y) == TARGET_LEN:
        return y
    if len(y) < TARGET_LEN:
        if len(y) == 0:
            return np.zeros(TARGET_LEN, dtype=np.float32)
        repeats = int(np.ceil(TARGET_LEN / len(y)))
        return np.tile(y, repeats)[:TARGET_LEN]

    if train:
        start = int(RNG.integers(0, len(y) - TARGET_LEN + 1))
    else:
        start = max(0, (len(y) - TARGET_LEN) // 2)
    return y[start : start + TARGET_LEN]


def normalize_audio(y):
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


def load_audio(path, train=True):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    y = fix_length(y, train=train)
    return normalize_audio(y)


def main(limit=None):
    audio_root = DATA / "train_audio"
    if not audio_root.exists():
        raise FileNotFoundError("缺少 data/train_audio，请先完成 Kaggle 下载与解压")

    files = sorted(audio_root.rglob("*.ogg"))
    if limit:
        files = files[:limit]

    OUT.mkdir(parents=True, exist_ok=True)
    for src in tqdm(files, desc="preprocess"):
        rel = src.relative_to(audio_root).with_suffix(".wav")
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        y = load_audio(src, train=True)
        sf.write(dst, y, TARGET_SR)

    print(f"processed={len(files)} out={OUT}")


if __name__ == "__main__":
    main()
