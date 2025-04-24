"""
Configuration settings for the application.
"""
import os
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    # Base settings
    APP_NAME: str = "Gesture & Face Authentication API"
    DEBUG: bool = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Database settings
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./app.db")
    
    # Face recognition settings
    FACE_RECOGNITION_MODEL: str = os.environ.get("FACE_RECOGNITION_MODEL", "hog")  # 'hog' or 'cnn'
    FACE_RECOGNITION_TOLERANCE: float = float(os.environ.get("FACE_RECOGNITION_TOLERANCE", "0.6"))
    
    # Gesture recognition settings
    GESTURE_CONFIDENCE_THRESHOLD: float = float(os.environ.get("GESTURE_CONFIDENCE_THRESHOLD", "0.6"))
    
    # Training settings
    MODEL_TRAINING_EPOCHS: int = int(os.environ.get("MODEL_TRAINING_EPOCHS", "60"))
    MODEL_TRAINING_BATCH_SIZE: int = int(os.environ.get("MODEL_TRAINING_BATCH_SIZE", "64"))
    MODEL_TRAINING_LEARNING_RATE: float = float(os.environ.get("MODEL_TRAINING_LEARNING_RATE", "0.001"))
    
    # Augmentation settings
    AUGMENTATION_SAMPLES: int = int(os.environ.get("AUGMENTATION_SAMPLES", "10"))
    
    # Path settings
    DATA_DIR: str = "data"
    MODEL_DIR: str = "models"
    UPLOAD_DIR: str = "uploads"
    FACES_DIR: str = os.path.join(UPLOAD_DIR, "faces")
    GESTURES_DIR: str = os.path.join(UPLOAD_DIR, "gestures")
    
    class Config:
        env_file = ".env"

# Create settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.MODEL_DIR, settings.UPLOAD_DIR, 
                 settings.FACES_DIR, settings.GESTURES_DIR]:
    os.makedirs(directory, exist_ok=True)
