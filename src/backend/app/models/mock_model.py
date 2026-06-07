import base64
import hashlib
import math
import random

from app.data.species import list_species


class MockBirdModel:
    def __init__(self, model_id, name, kind, description, bias=0):
        self.model_id = model_id
        self.name = name
        self.kind = kind
        self.description = description
        self.bias = bias

    def describe(self):
        return {
            "id": self.model_id,
            "name": self.name,
            "kind": self.kind,
            "loaded": True,
            "description": self.description,
        }

    def predict(self, audio_bytes, audio_info):
        seed = self._seed(audio_bytes, audio_info)
        rng = random.Random(seed)
        species = list_species()
        candidates = species[:]
        rng.shuffle(candidates)

        scores = []
        base = 0.78 + rng.random() * 0.12
        for index, item in enumerate(candidates[:5]):
            confidence = max(0.05, base - index * (0.08 + rng.random() * 0.03))
            scores.append(
                {
                    "species_id": item["id"],
                    "common_name": item["common_name"],
                    "scientific_name": item["scientific_name"],
                    "confidence": round(confidence, 4),
                    "rank": index + 1,
                }
            )

        return {
            "top5": scores,
            "waveform": self._waveform(audio_bytes, audio_info, rng),
            "mel_spectrogram": self._mel_placeholder(audio_bytes, rng),
        }

    def _seed(self, audio_bytes, audio_info):
        head = audio_bytes[:4096]
        text = f"{self.model_id}|{self.bias}|{audio_info.get('filename')}|{audio_info.get('bytes')}"
        digest = hashlib.sha256(head + text.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _waveform(self, audio_bytes, audio_info, rng):
        source = audio_info.get("waveform_points")
        if source:
            return {"sample_count": len(source), "points": source}

        points = []
        chunk_count = 96
        chunk_size = max(1, len(audio_bytes) // chunk_count)
        for index in range(chunk_count):
            chunk = audio_bytes[index * chunk_size : (index + 1) * chunk_size]
            if chunk:
                avg = sum(chunk) / len(chunk)
                value = (avg - 127.5) / 127.5
            else:
                value = math.sin(index / 8) * 0.2 + rng.uniform(-0.05, 0.05)
            points.append(round(max(-1, min(1, value)), 4))

        return {"sample_count": len(points), "points": points}

    def _mel_placeholder(self, audio_bytes, rng):
        rows = 24
        cols = 48
        matrix = []
        for row in range(rows):
            line = []
            for col in range(cols):
                base = math.sin((row + 1) * 0.23) * math.cos((col + 1) * 0.17)
                noise = rng.random() * 0.35
                value = max(0, min(1, 0.45 + base * 0.35 + noise))
                line.append(round(value, 3))
            matrix.append(line)

        svg = self._matrix_svg(matrix)
        image_base64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return {
            "type": "mock-mel-spectrogram",
            "mime": "image/svg+xml",
            "image_base64": image_base64,
            "matrix": matrix,
        }

    def _matrix_svg(self, matrix):
        cell = 8
        width = len(matrix[0]) * cell
        height = len(matrix) * cell
        rects = []
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                hue = 150 - int(value * 110)
                lightness = 18 + int(value * 55)
                color = f"hsl({hue}, 85%, {lightness}%)"
                rects.append(
                    f'<rect x="{x * cell}" y="{y * cell}" width="{cell}" height="{cell}" fill="{color}"/>'
                )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            f'<rect width="100%" height="100%" fill="#07130d"/>'
            f'{"".join(rects)}</svg>'
        )
