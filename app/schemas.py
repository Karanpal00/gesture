"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import datetime

# User schemas
class UserCreate(BaseModel):
    username: str
    email: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime.datetime
    
    class Config:
        orm_mode = True

class AuthResponse(BaseModel):
    message: str
    user_id: int

class FaceAuthRequest(BaseModel):
    username: str
    face_image_base64: str

# Gesture schemas
class GestureCreateRequest(BaseModel):
    user_id: int
    label: str
    binding: str
    keypoints: List[float]

class GestureRecognitionRequest(BaseModel):
    keypoints: List[float]

class GestureRecognitionResponse(BaseModel):
    gesture: str
    confidence: float
    binding: Optional[str] = None

class TrainingResponse(BaseModel):
    success: bool
    message: str
    training_started: bool

class TrainingStatusResponse(BaseModel):
    id: int
    status: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    accuracy: Optional[float] = None
    error_message: Optional[str] = None
    num_samples: Optional[int] = None
