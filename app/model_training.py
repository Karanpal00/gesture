"""
Model training functionality for gesture recognition.
"""
import pathlib
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
from typing import Dict, Any
from app.config import settings

def train_model(epochs: int = None, batch_size: int = None, lr: float = None) -> Dict[str, Any]:
    """
    Train a gesture recognition model using PyTorch.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        Dictionary with training results and status
    """
    # Use settings if arguments are not provided
    if epochs is None:
        epochs = settings.MODEL_TRAINING_EPOCHS
    if batch_size is None:
        batch_size = settings.MODEL_TRAINING_BATCH_SIZE
    if lr is None:
        lr = settings.MODEL_TRAINING_LEARNING_RATE
    
    # Where we look for gesture CSVs:
    data_dir = pathlib.Path(settings.DATA_DIR)
    # Where we dump models & metadata:
    model_dir = pathlib.Path(settings.MODEL_DIR)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load data
        X, y, lbl2id, binding_map = load_data(data_dir)
        
        if X is None:
            logging.warning("No CSV gesture data found - skipping training.")
            return {
                "success": False,
                "error": "No gesture data found for training."
            }
        
        # Split & scale
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        
        # To torch
        Xtr = torch.tensor(Xtr, dtype=torch.float32)
        ytr = torch.tensor(ytr, dtype=torch.long)
        Xte = torch.tensor(Xte, dtype=torch.float32)
        yte = torch.tensor(yte, dtype=torch.long)
        
        # Define model
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
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), batch_size):
                idx = perm[i : i + batch_size]
                out = model(Xtr[idx])
                loss = criterion(out, ytr[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0 or epoch == epochs:
                with torch.no_grad():
                    val_acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
                logging.info(f"epoch {epoch}/{epochs} — val acc: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
        
        # Export
        pt_path = model_dir / "gesture_clf_pt.pt"
        onnx_path = model_dir / "gesture_clf_pt.onnx"
        meta_path = model_dir / "meta_pt.pkl"
        
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
        
        logging.info("✅ Training complete. Saved model files.")
        
        return {
            "success": True,
            "accuracy": best_val_acc,
            "num_samples": len(X),
            "num_classes": len(lbl2id),
            "classes": list(lbl2id.keys())
        }
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def load_data(data_dir):
    """
    Load gesture data from CSV files.
    
    Args:
        data_dir: Directory containing gesture CSV files
        
    Returns:
        Tuple of (X, y, label_map, binding_map) or (None, None, None, None) if no data
    """
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        # No data yet
        return None, None, None, None
    
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            logging.error(f"Error reading CSV file {f}: {str(e)}")
    
    if not frames:
        return None, None, None, None
    
    df = pd.concat(frames, ignore_index=True)
    
    # Extract features and labels
    X = df.iloc[:, 2:].values.astype(np.float32)
    y_labels = df["label"].values
    
    # Create label map
    classes = np.unique(y_labels)
    lbl2id = {c: i for i, c in enumerate(classes)}
    y_num = np.vectorize(lbl2id.get)(y_labels)
    
    # Create binding map
    binding_map = {row["label"]: row["binding"] for _, row in df.iterrows()}
    
    return X, y_num, lbl2id, binding_map
