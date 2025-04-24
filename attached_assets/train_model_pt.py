#!/usr/bin/env python
"""
Train a lightweight MLP on hand‑keypoint vectors with PyTorch,
export to ONNX, and save label map + normaliser + binding map.

This defines a function `train_model_pt()` that your FastAPI app
can import and call on demand. You can still run it directly:

    python scripts/train_model_pt.py
"""

import pathlib
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Where we look for gesture CSVs:
DATA_DIR = pathlib.Path("data")
# Where we dump models & metadata:
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        # no data yet, let the caller decide what to do
        return None, None, None, None

    frames = [pd.read_csv(f) for f in csvs]
    df = pd.concat(frames, ignore_index=True)
    X = df.iloc[:, 2:].values.astype(np.float32)
    y_labels = df["label"].values

    classes = np.unique(y_labels)
    lbl2id = {c: i for i, c in enumerate(classes)}
    y_num = np.vectorize(lbl2id.get)(y_labels)

    binding_map = {row["label"]: row["binding"] for _, row in df.iterrows()}

    return X, y_num, lbl2id, binding_map


def train_model_pt(epochs=60, batch_size=64, lr=1e-3):
    """Load data, train MLP, export models + metadata."""
    X, y, lbl2id, binding_map = load_data()
    if X is None:
        # no data to train on
        print("⚠️  No CSV gesture data found in data/processed/ — skipping training.")
        return

    # split & scale
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # to torch
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

    # define model
    class KeypointMLP(nn.Module):
        def __init__(self, in_dim=X.shape[1], n_classes=len(lbl2id)):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, n_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = KeypointMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i : i + batch_size]
            out = model(Xtr[idx])
            loss = criterion(out, ytr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                val_acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
            print(f"epoch {epoch}/{epochs} — val acc: {val_acc:.2f}%")

    # export
    pt_path   = MODEL_DIR / "gesture_clf_pt.pt"
    onnx_path = MODEL_DIR / "gesture_clf_pt.onnx"
    meta_path = MODEL_DIR / "meta_pt.pkl"

    torch.save(model.state_dict(), pt_path)

    dummy = torch.randn(1, X.shape[1])
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["float_input"],
        output_names=["output"],
        dynamic_axes={"float_input": {0: "batch"}},
    )

    with open(meta_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "label_map": lbl2id,
            "binding_map": binding_map
        }, f)

    print("✅ Training complete. Saved:")
    print(f"   • {pt_path}")
    print(f"   • {onnx_path}")
    print(f"   • {meta_path}")


if __name__ == "__main__":
    train_model_pt()
