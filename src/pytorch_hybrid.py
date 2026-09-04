from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(r"E:\Terraspectra")
DATA_DIR = PROJECT_ROOT / "data" / "patches_shared_pca"
MODEL_PATH = PROJECT_ROOT / "models" / "pytorch_hybrid_baseline.pt"

NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 1
SEED = 42


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=32, heads=2, feed_forward_dim=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feed_forward_dim, embed_dim),
        )

    def forward(self, inputs):
        normalized = self.norm1(inputs)
        attended, _ = self.attention(normalized, normalized, normalized)
        x = inputs + attended
        return x + self.feed_forward(self.norm2(x))


class PyTorchHybrid(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AvgPool3d((2, 2, 1)),
        )
        self.transformer = TransformerBlock()
        self.classifier = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, inputs):
        features = self.cnn(inputs)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        return self.classifier(tokens.mean(dim=1))


def load_split(split_name):
    split_dir = DATA_DIR / split_name
    features = np.load(split_dir / f"X_{split_name}.npy").astype(np.float32)
    labels = np.load(split_dir / f"y_{split_name}.npy").astype(np.int64)
    if features.shape[1:] != (32, 32, 30):
        raise ValueError(f"Unexpected {split_name} shape: {features.shape}")
    features = torch.from_numpy(features).permute(0, 3, 1, 2).unsqueeze(1)
    return features, torch.from_numpy(labels)


def run_epoch(model, loader, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    for features, labels in loader:
        if training:
            optimizer.zero_grad()
        logits = model(features)
        loss = loss_function(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += len(labels)
    return total_loss / samples, correct / samples


def main():
    torch.manual_seed(SEED)
    train_features, train_labels = load_split("train")
    val_features, val_labels = load_split("val")
    train_loader = DataLoader(TensorDataset(train_features, train_labels), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_labels), BATCH_SIZE)

    model = PyTorchHybrid()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_function, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_function)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved_model={MODEL_PATH}")


if __name__ == "__main__":
    main()
