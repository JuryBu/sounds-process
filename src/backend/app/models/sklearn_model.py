from pathlib import Path

import joblib
import numpy as np

from app.data.species import species_label
from app.services.audio_features import extract_feature_vector


ROOT = Path(__file__).resolve().parents[4]


class SklearnBirdModel:
    def __init__(self, model_id, name, kind, description, model_path):
        self.model_id = model_id
        self.name = name
        self.kind = kind
        self.description = description
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"缺少真实模型文件：{self.model_path}")
        payload = joblib.load(self.model_path)
        self.model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.label_encoder = payload.get("label_encoder") if isinstance(payload, dict) else None

    def describe(self):
        return {
            "id": self.model_id,
            "name": self.name,
            "kind": self.kind,
            "loaded": True,
            "description": self.description,
        }

    def predict(self, audio_bytes, audio_info):
        vector = extract_feature_vector(audio_bytes, audio_info.get("filename") or "audio.ogg", skip_pitch=True)
        x = vector.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(x)[0], dtype=np.float64)
        else:
            predicted = self.model.predict(x)[0]
            classes = getattr(self.model, "classes_", np.asarray([predicted]))
            proba = np.zeros(len(classes), dtype=np.float64)
            proba[int(np.where(classes == predicted)[0][0])] = 1.0

        class_labels = self._classes()
        top_indices = np.argsort(proba)[::-1][:5]
        top5 = []
        for rank, index in enumerate(top_indices, start=1):
            label = str(class_labels[index])
            species = species_label(label)
            top5.append(
                {
                    "species_id": species["id"],
                    "common_name": species["common_name"],
                    "scientific_name": species["scientific_name"],
                    "confidence": round(float(proba[index]), 4),
                    "rank": rank,
                }
            )
        return {
            "top5": top5,
            "waveform": {"sample_count": 0, "points": []},
            "mel_spectrogram": {"type": "pending", "mime": "image/svg+xml", "image_base64": "", "matrix": []},
        }

    def _classes(self):
        classes = getattr(self.model, "classes_", None)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(classes)
        return classes
