import json
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "logs"
SR = 32000
TARGET_SECONDS = 5
TARGET_LEN = SR * TARGET_SECONDS


def time_shift(y, max_shift=6400, rng=None):
    rng = rng or np.random.default_rng()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(y, shift)


def add_noise_snr(y, snr_db=15, rng=None):
    rng = rng or np.random.default_rng()
    signal_power = float(np.mean(y**2))
    if signal_power <= 1e-12:
        return y.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def spec_augment(mel, freq_mask=30, time_mask=40, rng=None):
    rng = rng or np.random.default_rng()
    out = mel.copy()
    if out.shape[0] > 1:
        width = int(rng.integers(1, min(freq_mask, out.shape[0]) + 1))
        start = int(rng.integers(0, out.shape[0] - width + 1))
        out[start : start + width, :] = out.mean()
    if out.shape[1] > 1:
        width = int(rng.integers(1, min(time_mask, out.shape[1]) + 1))
        start = int(rng.integers(0, out.shape[1] - width + 1))
        out[:, start : start + width] = out.mean()
    return out


def time_stretch_fixed(y, rate):
    stretched = librosa.effects.time_stretch(y, rate=rate)
    if len(stretched) < TARGET_LEN:
        repeats = int(np.ceil(TARGET_LEN / max(1, len(stretched))))
        stretched = np.tile(stretched, repeats)
    return stretched[:TARGET_LEN].astype(np.float32)


def pitch_shift(y, semitones):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=semitones).astype(np.float32)


def smoke_test():
    rng = np.random.default_rng(2026)
    t = np.arange(TARGET_LEN) / SR
    y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
    payload = {
        "input_shape": list(y.shape),
        "time_shift_shape": list(time_shift(y, rng=rng).shape),
        "noise_shape": list(add_noise_snr(y, rng=rng).shape),
        "specaug_shape": list(spec_augment(mel, rng=rng).shape),
        "stretch_shape": list(time_stretch_fixed(y, rate=1.1).shape),
        "pitch_shape": list(pitch_shift(y, semitones=1).shape),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "augment_smoke.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    smoke_test()
