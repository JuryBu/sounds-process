# BirdVoice Backend

FastAPI 后端，默认加载 `models/xgboost.joblib` 模型。
后端会按本地存在的模型文件自动注册 KNN、SVM 等 `sklearn` 基线；mock 联调模型仅在设置 `BIRDVOICE_ENABLE_MOCKS=1` 时用于界面联调。

## 安装

```bash
cd 大作业/src/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果只在 `src/backend/` 下单独安装依赖，也需要安装根目录 `requirements.txt` 中的 `librosa`、`numpy`、`scikit-learn`、`xgboost` 与 `joblib`，推理会复用特征提取代码。

## 启动

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8017 --reload
```

启动后访问：

- `GET http://127.0.0.1:8017/api/health`
- `GET http://127.0.0.1:8017/api/models`
- `POST http://127.0.0.1:8017/api/predict`
- `GET http://127.0.0.1:8017/api/species/sp001`

## 预测接口示例

```bash
curl -X POST "http://127.0.0.1:8017/api/predict" ^
  -F "model_name=xgboost-real" ^
  -F "file=@demo.ogg"
```

`/api/predict` 会读取上传音频并提取 MFCC、Mel、能量与频谱统计特征。默认模型为 `xgboost-real`；缺少对应模型文件时该模型不会注册，接口会返回当前可用的模型列表。

## 模型接入位置

- `app/models/registry.py`：模型注册、懒加载和切换入口
- `app/models/sklearn_model.py`：真实 `sklearn` 模型加载与 Top-5 推理
- `app/models/mock_model.py`：mock 推理逻辑，仅在 `BIRDVOICE_ENABLE_MOCKS=1` 时用于界面联调
- `app/services/audio_info.py`：上传音频基础解析和波形采样
- `app/services/audio_features.py`：后端在线特征提取
- `app/data/species.py`：优先从 `data/taxonomy.csv` 加载物种资料
