import os

from app.models.mock_model import MockBirdModel
from app.models.sklearn_model import ROOT, SklearnBirdModel


class ModelRegistry:
    def __init__(self):
        self.default_name = ""
        self._items = {}
        self._loaded = {}

    def register(self, model_id, name, kind, description, factory):
        self._items[model_id] = {
            "id": model_id,
            "name": name,
            "kind": kind,
            "description": description,
            "factory": factory,
        }

    def get(self, model_id):
        if model_id not in self._items:
            raise KeyError(model_id)
        if model_id not in self._loaded:
            self._loaded[model_id] = self._items[model_id]["factory"]()
        return self._loaded[model_id]

    def loaded_names(self):
        return list(self._loaded.keys())

    def unload(self, model_id=None):
        if model_id is None:
            names = list(self._loaded.keys())
            self._loaded.clear()
            return names
        if model_id not in self._items:
            raise KeyError(model_id)
        existed = model_id in self._loaded
        self._loaded.pop(model_id, None)
        return [model_id] if existed else []

    def describe_all(self):
        rows = []
        for model_id, item in self._items.items():
            rows.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "kind": item["kind"],
                    "loaded": model_id in self._loaded,
                    "description": item["description"],
                }
            )
        return rows

    def real_model_names(self):
        return [model_id for model_id, item in self._items.items() if item["kind"] == "sklearn"]


def register_sklearn_model(model_id, name, filename, description):
    model_path = ROOT / "models" / filename
    if not model_path.exists():
        return
    model_registry.register(
        model_id,
        name,
        "sklearn",
        description,
        lambda model_id=model_id, name=name, description=description, model_path=model_path: SklearnBirdModel(
            model_id,
            name,
            "sklearn",
            description,
            model_path,
        ),
    )
    if not model_registry.default_name:
        model_registry.default_name = model_id


model_registry = ModelRegistry()

register_sklearn_model(
    "xgboost-real",
    "XGBoost Real Best",
    "xgboost.joblib",
    "真实 XGBoost 最佳传统机器学习模型，使用 BirdCLEF2026 统计音频特征训练，当前验证集 Accuracy=0.2711、Top-5=0.4972。",
)

register_sklearn_model(
    "knn-real",
    "KNN Real Baseline",
    "knn.joblib",
    "真实 KNN 基线模型，使用 BirdCLEF2026 统计音频特征训练，用于与 XGBoost 横向对比。",
)

register_sklearn_model(
    "svm-real",
    "SVM Real Baseline",
    "svm.joblib",
    "真实 SVM 基线模型，使用 BirdCLEF2026 统计音频特征训练；仅作为弱基线对照。",
)

if os.getenv("BIRDVOICE_ENABLE_MOCKS") == "1":
    model_registry.register(
        "mock-cnn",
        "Mock CNN Baseline",
        "mock",
        "模拟 CNN 鸟声识别模型，仅在 BIRDVOICE_ENABLE_MOCKS=1 时用于界面联调。",
        lambda: MockBirdModel(
            "mock-cnn",
            "Mock CNN Baseline",
            "mock",
            "模拟 CNN 鸟声识别模型，仅用于界面联调。",
            bias=11,
        ),
    )
    model_registry.register(
        "mock-ensemble",
        "Mock Ensemble",
        "mock",
        "模拟集成模型，仅在 BIRDVOICE_ENABLE_MOCKS=1 时用于界面联调。",
        lambda: MockBirdModel(
            "mock-ensemble",
            "Mock Ensemble",
            "mock",
            "模拟集成模型，仅用于界面联调。",
            bias=37,
        ),
    )
    if not model_registry.default_name:
        model_registry.default_name = "mock-cnn"
