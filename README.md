# 基于声音的鸟类识别系统

语音识别技术课程大作业，用 BirdCLEF2026 数据做鸟鸣分类。整个项目从数据处理到特征提取再到模型训练和前端演示都做了，后端用 FastAPI，前端 Vue3。

## 目录

```
src/data/       数据划分、预处理
src/features/   特征提取
src/training/   模型训练
src/backend/    FastAPI 后端
src/frontend/   Vue3 前端
models/         训练好的模型
data/           标签映射、示例音频
```

## 环境

Python 3.9+，前端需要 Node.js。

```bash
pip install -r requirements.txt
cd src/frontend && npm install
```

## 运行

```bash
.\run_dev.ps1
```

浏览器打开 http://127.0.0.1:5177 ，在 data/demo_audio/ 下有示例音频可以直接上传试。

停止用 `.\stop_dev.ps1`

## 训练

需要先从 Kaggle 下载 BirdCLEF2026 数据集解压到 data/ 下面。仓库里只有示例音频和标签文件，跑演示不需要完整数据集。

```bash
python src/data/make_splits.py
python src/features/extract_features.py
python src/training/train_ml_baselines.py --folds 5
python src/training/train_cnn_models.py --model resnet50 --epochs 10
python src/training/ensemble_and_analysis.py
```

## 结果

传统模型在验证集上的表现：

| 模型 | Accuracy | Macro F1 | Top-5 |
|------|----------|----------|-------|
| XGBoost | 0.2711 | 0.2071 | 0.4972 |
| Random Forest | 0.2542 | 0.1585 | 0.4508 |
| KNN | 0.0568 | 0.0481 | 0.1355 |
| SVM | 0.0011 | 0.0006 | 0.0104 |

深度模型用的是 4096 样本子集训练 10 轮，和传统模型不是同口径：

| 模型 | Val Acc | Top-5 |
|------|---------|-------|
| ResNet-50 | 0.3329 | 0.4512 |
| Bi-LSTM | 0.2366 | 0.4268 |
| EfficientNet-B2 | 0.2122 | 0.3695 |
| Custom CNN | 0.1780 | 0.3634 |
| AST-like | 0.1427 | 0.3415 |
| Mamba-like | 0.1073 | 0.2720 |

206 类长尾分布比较严重，Macro F1 比 Accuracy 更能说明问题。

## 其他

完整数据集大概 15GB，深度模型权重也比较大，都没放进仓库。演示用的 joblib 模型在 models/ 里，clone 下来就能跑。
