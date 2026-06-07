# 基于声音的鸟类识别系统

语音识别技术课程大作业。基于 BirdCLEF2026 数据集，实现一个鸟鸣音频分类系统，包含音频特征提取、传统机器学习与深度学习模型训练，以及一个用于演示的 Web 界面。

## 主要功能

- **特征提取**：MFCC 及其一阶差分、RMS 能量、频谱质心/带宽/滚降、过零率、音高等统计特征（共 174 维），以及用于深度模型的 128×313 Mel 频谱图。
- **传统模型**：KNN、SVM、随机森林、XGBoost。
- **深度模型**：自定义 CNN、ResNet-50、EfficientNet-B2、Bi-LSTM、AST 风格 Transformer、Mamba 风格序列模型。
- **模型融合**：加权平均、多数投票、Stacking。
- **Web 演示**：上传或录制音频，查看波形、Mel 频谱图与 Top-5 识别结果，支持多模型对比。

## 目录结构

```
src/data/       数据集划分、预处理、EDA、增强
src/features/   音频特征提取
src/training/   传统模型、深度模型训练与集成分析
src/backend/    FastAPI 推理后端
src/frontend/   Vue 3 演示前端
models/         训练得到的模型（仓库内含演示用 joblib）
data/           taxonomy、标签映射、示例音频
```

## 环境

Python 3.11，安装依赖：

```bash
pip install -r requirements.txt
```

前端需要 Node.js：

```bash
cd src/frontend
npm install
```

## 数据准备

从 Kaggle 下载 [BirdCLEF2026](https://www.kaggle.com/competitions/birdclef-2026) 数据集，解压到 `data/` 目录，使其包含 `train_audio/`、`train.csv`、`taxonomy.csv` 等。仓库已包含 `data/taxonomy.csv`、`data/label_map.json` 与示例音频，仅运行演示无需下载完整数据集。

## 训练流程

```bash
# 划分数据集（按类别分层）
python src/data/make_splits.py

# 提取特征
python src/features/extract_features.py

# 训练传统模型
python src/training/train_ml_baselines.py --folds 5

# 训练深度模型（以 ResNet-50 为例）
python src/training/train_cnn_models.py --model resnet50 --epochs 10

# 模型融合与混淆分析
python src/training/ensemble_and_analysis.py
```

## 运行演示

后端默认加载 `models/xgboost.joblib`，前端为 Vue 3 + Vite。

```bash
# 一键启动后端(8017)与前端(5177)
.\run_dev.ps1
```

浏览器打开 http://127.0.0.1:5177 ，上传 `data/demo_audio/` 下的示例音频即可看到识别结果。停止服务：

```bash
.\stop_dev.ps1
```

也可分别启动：

```bash
cd src/backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8017
```

## 实验结果（验证集）

传统模型：

| 模型 | Accuracy | Macro F1 | Top-5 Accuracy |
|------|----------|----------|----------------|
| XGBoost | 0.2711 | 0.2071 | 0.4972 |
| 随机森林 | 0.2542 | 0.1585 | 0.4508 |
| KNN | 0.0568 | 0.0481 | 0.1355 |
| SVM (RBF 近似) | 0.0011 | 0.0006 | 0.0104 |

深度模型（各训练 10 轮，取验证集最优）：

| 模型 | Best Val Accuracy | Best Top-5 |
|------|-------------------|------------|
| ResNet-50 | 0.3329 | 0.4512 |
| Bi-LSTM | 0.2366 | 0.4268 |
| EfficientNet-B2 | 0.2122 | 0.3695 |
| 自定义 CNN | 0.1780 | 0.3634 |
| AST 风格 Transformer | 0.1427 | 0.3415 |
| Mamba 风格 | 0.1073 | 0.2720 |

数据集类别分布长尾明显，Macro F1 比整体 Accuracy 更能反映少样本类别的识别质量。

## 说明

原始数据集（约 15GB）与深度模型权重（`.pth`）体积较大，未包含在仓库中。演示用的 XGBoost / SVM / KNN 模型（joblib）已包含，克隆后可直接启动后端体验识别功能。
