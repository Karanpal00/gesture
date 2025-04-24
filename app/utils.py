"""
Utility functions for the application.
"""
import os
import pickle
import logging
import numpy as np
import onnxruntime
from typing import Tuple, Dict, Any, Optional

def load_model_and_metadata() -> Tuple[Optional[onnxruntime.InferenceSession], Optional[Dict[str, Any]]]:
    """
    Load the gesture recognition model and metadata.
    
    Returns:
        Tuple of (onnx_model, metadata) or (None, None) if files are not found
    """
    onnx_path = os.path.join("models", "gesture_clf_pt.onnx")
    meta_path = os.path.join("models", "meta_pt.pkl")
    
    # Check if files exist
    if not os.path.exists(onnx_path) or not os.path.exists(meta_path):
        logging.warning(f"Model files not found: {onnx_path} or {meta_path}")
        return None, None
    
    try:
        # Load ONNX model
        onnx_model = onnxruntime.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Load metadata
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        return onnx_model, metadata
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)

def clean_filename(filename: str) -> str:
    """
    Clean a filename to ensure it's safe for the filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Replace unsafe characters
    unsafe_chars = [" ", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    
    return filename
