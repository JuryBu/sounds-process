import base64
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DATA_SRC = ROOT / "src" / "data"
FEATURE_SRC = ROOT / "src" / "features"
for path in (DATA_SRC, FEATURE_SRC):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from preprocess import TARGET_SR, fix_length, normalize_audio  # noqa: E402
from extract_features import feature_vector, mel_image  # noqa: E402


def extract_audio_features(audio_bytes, filename, skip_pitch=True):
    y, sr = _load_audio_bytes(audio_bytes, filename)
    duration_seconds = round(float(len(y) / sr), 4) if sr and len(y) else None
    fixed = normalize_audio(fix_length(y, train=False))
    mel = mel_image(fixed)
    vector = feature_vector(fixed, skip_pitch=skip_pitch)
    waveform = _sample_waveform(fixed)
    return {
        "sample_rate": sr,
        "target_sample_rate": TARGET_SR,
        "duration_seconds": duration_seconds,
        "processed_duration_seconds": round(float(len(fixed) / TARGET_SR), 4) if TARGET_SR else None,
        "feature_vector_shape": list(vector.shape),
        "mel": _mel_payload(mel),
        "waveform": {
            "sample_count": len(waveform),
            "points": waveform,
        },
    }


def extract_feature_vector(audio_bytes, filename, skip_pitch=True):
    y, _ = _load_audio_bytes(audio_bytes, filename)
    fixed = normalize_audio(fix_length(y, train=False))
    return feature_vector(fixed, skip_pitch=skip_pitch)


def _load_audio_bytes(audio_bytes, filename):
    import librosa

    suffix = Path(filename or "audio.wav").suffix or ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            temp_path = handle.name
        y, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
    return y.astype(np.float32), sr


def _sample_waveform(y, target_count=140):
    if len(y) == 0:
        return []
    step = max(1, len(y) // target_count)
    points = []
    for start in range(0, len(y), step):
        chunk = y[start : start + step]
        if len(chunk) == 0:
            continue
        peak = float(chunk[np.argmax(np.abs(chunk))])
        points.append(round(max(-1.0, min(1.0, peak)), 4))
        if len(points) >= target_count:
            break
    return points


def _mel_payload(mel):
    normalized = _normalize_matrix(mel)
    thumbnail = _downsample_matrix(normalized, rows=48, cols=96)
    svg = _matrix_svg(thumbnail)
    return {
        "type": "mel-spectrogram",
        "mime": "image/svg+xml",
        "image_base64": base64.b64encode(svg.encode("utf-8")).decode("ascii"),
        "matrix": [[round(float(value), 4) for value in row] for row in thumbnail],
    }


def _normalize_matrix(matrix):
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, neginf=0.0, posinf=0.0)
    lo = float(np.min(matrix))
    hi = float(np.max(matrix))
    if hi - lo < 1e-8:
        return np.zeros_like(matrix)
    return (matrix - lo) / (hi - lo)


def _downsample_matrix(matrix, rows, cols):
    row_index = np.linspace(0, matrix.shape[0] - 1, rows).astype(int)
    col_index = np.linspace(0, matrix.shape[1] - 1, cols).astype(int)
    return matrix[row_index][:, col_index]


def _matrix_svg(matrix):
    cell_w = 10
    cell_h = 5
    height = len(matrix) * cell_h
    width = len(matrix[0]) * cell_w if len(matrix) else 0
    rects = []
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            color = _viridis_like(value)
            rects.append(
                f'<rect x="{x * cell_w}" y="{y * cell_h}" width="{cell_w}" '
                f'height="{cell_h}" fill="{color}"/>'
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<rect width="100%" height="100%" fill="#07030f"/>'
        f'{"".join(rects)}</svg>'
    )


def _viridis_like(value):
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    value = max(0.0, min(1.0, float(value)))
    scaled = value * (len(stops) - 1)
    idx = min(len(stops) - 2, int(scaled))
    mix = scaled - idx
    a = stops[idx]
    b = stops[idx + 1]
    rgb = [round(a[i] + (b[i] - a[i]) * mix) for i in range(3)]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
