from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "features" / "mel"
MODELS = ROOT / "models"


class TinyCnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)


def main(limit=64):
    files = sorted(FEATURES.glob("*.npy"))[:limit]
    if not files:
        raise FileNotFoundError("缺少 features/mel/*.npy，请先运行特征提取")
    x = np.stack([np.load(p) for p in files]).astype("float32")
    x = (x - x.mean()) / (x.std() + 1e-6)
    labels = np.arange(len(files)) % min(8, len(files))
    ds = TensorDataset(torch.from_numpy(x[:, None, :, :]), torch.from_numpy(labels).long())
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCnn(num_classes=int(labels.max()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "tiny_cnn_smoke.pth")
    print(f"device={device} loss={float(loss.detach().cpu()):.4f}")


if __name__ == "__main__":
    main()
