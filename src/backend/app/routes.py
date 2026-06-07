from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.data.species import get_species, list_species
from app.models.registry import model_registry
from app.schemas import HealthResponse, PredictResponse
from app.services.audio_features import extract_audio_features
from app.services.audio_info import read_audio_info


api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "service": "BirdVoice backend",
        "version": "0.1.0",
        "loaded_models": model_registry.loaded_names(),
    }


@api_router.get("/models")
def models():
    default_model = model_registry.default_name
    return {
        "default_model": default_model,
        "models": model_registry.describe_all(),
    }


@api_router.post("/predict", response_model=PredictResponse)
async def predict(
    model_name: str = Form(default=""),
    file: UploadFile = File(...),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        model = model_registry.get(model_name or model_registry.default_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"模型不存在：{model_name}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"模型加载失败：{exc}")

    audio_info = read_audio_info(audio_bytes, file.filename or "audio")
    result = model.predict(audio_bytes, audio_info)
    try:
        features = extract_audio_features(audio_bytes, file.filename or "audio")
        result["waveform"] = features["waveform"]
        result["mel_spectrogram"] = features["mel"]
        audio_info["sample_rate"] = audio_info.get("sample_rate") or features["sample_rate"]
        audio_info["duration_seconds"] = audio_info.get("duration_seconds") or features["duration_seconds"]
        audio_info["processed_duration_seconds"] = features["processed_duration_seconds"]
        audio_info["feature_vector_shape"] = features["feature_vector_shape"]
    except Exception as exc:
        audio_info["feature_error"] = str(exc)

    return {
        "model": model.describe(),
        "filename": file.filename,
        "audio": audio_info,
        "top5": result["top5"],
        "waveform": result["waveform"],
        "mel_spectrogram": result["mel_spectrogram"],
        "note": "预测结果来自所选真实 sklearn 模型；默认使用验证集表现最好的 XGBoost 模型。",
    }


@api_router.post("/compare")
async def compare(
    model_names: str = Form(default=""),
    file: UploadFile = File(...),
):
    if not model_names:
        defaults = model_registry.real_model_names()[:3] or [model_registry.default_name]
        model_names = ",".join(dict.fromkeys(defaults))
    names = [name.strip() for name in model_names.split(",") if name.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="至少选择一个模型")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空")

    audio_info = read_audio_info(audio_bytes, file.filename or "audio")
    feature_payload = None
    feature_error = None
    try:
        feature_payload = extract_audio_features(audio_bytes, file.filename or "audio")
        audio_info["sample_rate"] = audio_info.get("sample_rate") or feature_payload["sample_rate"]
        audio_info["duration_seconds"] = audio_info.get("duration_seconds") or feature_payload["duration_seconds"]
        audio_info["processed_duration_seconds"] = feature_payload["processed_duration_seconds"]
        audio_info["feature_vector_shape"] = feature_payload["feature_vector_shape"]
    except Exception as exc:
        feature_error = str(exc)

    rows = []
    for name in names:
        try:
            model = model_registry.get(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"模型不存在：{name}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f"模型加载失败：{name}：{exc}")
        result = model.predict(audio_bytes, audio_info)
        rows.append(
            {
                "model": model.describe(),
                "top5": result["top5"],
                "winner": result["top5"][0] if result["top5"] else None,
            }
        )

    return {
        "filename": file.filename,
        "audio": audio_info,
        "results": rows,
        "mel_spectrogram": feature_payload["mel"] if feature_payload else None,
        "feature_error": feature_error,
        "note": "对比模式使用同一段音频分别调用当前可用的多个真实 sklearn 模型。",
    }


@api_router.delete("/models/cache")
def clear_model_cache():
    released = model_registry.unload()
    return {"released": released, "loaded_models": model_registry.loaded_names()}


@api_router.delete("/models/{model_name}/cache")
def clear_one_model_cache(model_name: str):
    try:
        released = model_registry.unload(model_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"模型不存在：{model_name}")
    return {"released": released, "loaded_models": model_registry.loaded_names()}


@api_router.get("/species")
def species_all():
    return {"items": list_species()}


@api_router.get("/species/{species_id}")
def species_detail(species_id: str):
    species = get_species(species_id)
    if not species:
        raise HTTPException(status_code=404, detail=f"未找到物种：{species_id}")
    return species
