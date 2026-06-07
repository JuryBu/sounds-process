import io
import struct
import wave


def read_audio_info(audio_bytes, filename):
    audio_format = _guess_format(filename, audio_bytes)
    info = {
        "filename": filename,
        "format": audio_format,
        "bytes": len(audio_bytes),
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
        "sample_width": None,
        "frames": None,
    }

    if audio_format == "wav":
        wav_info = _read_wav(audio_bytes)
        info.update(wav_info)

    return info


def _guess_format(filename, audio_bytes):
    lower_name = filename.lower()
    if lower_name.endswith(".wav") or audio_bytes[:4] == b"RIFF":
        return "wav"
    if lower_name.endswith(".ogg") or audio_bytes[:4] == b"OggS":
        return "ogg"
    if lower_name.endswith(".mp3") or audio_bytes[:3] == b"ID3":
        return "mp3"
    if lower_name.endswith(".flac") or audio_bytes[:4] == b"fLaC":
        return "flac"
    return "unknown"


def _read_wav(audio_bytes):
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            duration = frames / sample_rate if sample_rate else None
            raw = wav.readframes(frames)
            points = _sample_pcm(raw, sample_width, channels)
            return {
                "duration_seconds": round(duration, 4) if duration is not None else None,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "frames": frames,
                "waveform_points": points,
            }
    except wave.Error:
        return {}


def _sample_pcm(raw, sample_width, channels, target_count=120):
    values = _decode_pcm(raw, sample_width)
    if channels and channels > 1:
        values = _mix_channels(values, channels)

    if not values:
        return []

    step = max(1, len(values) // target_count)
    points = []
    for start in range(0, len(values), step):
        chunk = values[start : start + step]
        if not chunk:
            continue
        peak = max(chunk, key=lambda item: abs(item))
        points.append(round(max(-1, min(1, peak)), 4))
        if len(points) >= target_count:
            break
    return points


def _decode_pcm(raw, sample_width):
    if sample_width == 1:
        return [((byte - 128) / 128) for byte in raw]

    if sample_width == 2:
        count = len(raw) // 2
        samples = struct.unpack("<" + "h" * count, raw[: count * 2])
        return [sample / 32768 for sample in samples]

    if sample_width == 3:
        values = []
        for index in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[index : index + 3], "little", signed=True)
            values.append(value / 8388608)
        return values

    if sample_width == 4:
        count = len(raw) // 4
        samples = struct.unpack("<" + "i" * count, raw[: count * 4])
        return [sample / 2147483648 for sample in samples]

    return []


def _mix_channels(values, channels):
    mixed = []
    for index in range(0, len(values), channels):
        frame = values[index : index + channels]
        if frame:
            mixed.append(sum(frame) / len(frame))
    return mixed
