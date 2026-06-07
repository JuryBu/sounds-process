from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    loaded_models: List[str]


class ModelInfo(BaseModel):
    id: str
    name: str
    kind: str
    loaded: bool
    description: str


class AudioInfo(BaseModel):
    filename: str
    format: str
    bytes: int
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    sample_width: Optional[int] = None
    frames: Optional[int] = None
    feature_vector_shape: Optional[List[int]] = None
    feature_error: Optional[str] = None


class SpeciesScore(BaseModel):
    species_id: str
    common_name: str
    scientific_name: str
    confidence: float
    rank: int


class WaveformData(BaseModel):
    sample_count: int
    points: List[float]


class MelSpectrogramData(BaseModel):
    type: str
    mime: str
    image_base64: str
    matrix: List[List[float]]


class PredictResponse(BaseModel):
    model: ModelInfo
    filename: Optional[str]
    audio: AudioInfo
    top5: List[SpeciesScore]
    waveform: WaveformData
    mel_spectrogram: MelSpectrogramData
    note: str
