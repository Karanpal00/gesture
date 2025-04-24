"""
Gesture recognition and training endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import logging
import torch
import os
import json
import onnxruntime
import pandas as pd
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
import base64
import cv2
from datetime import datetime

from app.models import GestureData, TrainingSession, User, get_db
from app.schemas import (
    GestureCreateRequest, 
    GestureRecognitionRequest, 
    GestureRecognitionResponse, 
    TrainingResponse
)
from app.model_training import train_model
from app.data_augmentation import augment_gesture_data
from app.utils import load_model_and_metadata

router = APIRouter()

# Cache for model and metadata
model_cache = {"model": None, "metadata": None, "last_loaded": None}

async def load_or_get_model():
    """Load the model and metadata or get from cache."""
    # Check if model is loaded and recent
    if (model_cache["model"] is not None and model_cache["metadata"] is not None and
            model_cache["last_loaded"] is not None and
            (datetime.now() - model_cache["last_loaded"]).seconds < 300):  # 5 minutes cache
        return model_cache["model"], model_cache["metadata"]
    
    # Load the model and metadata
    try:
        model, metadata = load_model_and_metadata()
        model_cache["model"] = model
        model_cache["metadata"] = metadata
        model_cache["last_loaded"] = datetime.now()
        return model, metadata
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/recognize", response_model=GestureRecognitionResponse)
async def recognize_gesture(request: GestureRecognitionRequest):
    """
    Recognize a gesture from hand keypoints.
    
    Takes the hand keypoints from MediaPipe or similar and runs inference
    using the trained model.
    """
    try:
        # Load the model and metadata
        model, metadata = await load_or_get_model()
        
        if model is None or metadata is None:
            raise HTTPException(
                status_code=404,
                detail="No gesture model found. Please train a model first."
            )
        
        # Extract metadata components
        scaler = metadata["scaler"]
        label_map = metadata["label_map"]
        binding_map = metadata["binding_map"]
        
        # Create a mapping from label id to label name
        id_to_label = {v: k for k, v in label_map.items()}
        
        # Get keypoints from request
        keypoints = np.array(request.keypoints, dtype=np.float32)
        
        # Check if keypoints are valid
        if keypoints.size == 0:
            return {"gesture": "none", "confidence": 0.0, "binding": None}
        
        # Scale the keypoints
        keypoints_scaled = scaler.transform(keypoints.reshape(1, -1))
        
        # Create ONNX Runtime input
        ort_inputs = {model.get_inputs()[0].name: keypoints_scaled.astype(np.float32)}
        
        # Run inference
        ort_outputs = model.run(None, ort_inputs)
        
        # Get the softmax probabilities
        logits = ort_outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Get the predicted class and confidence
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        
        # Check confidence threshold
        if confidence < 0.6:  # Confidence threshold
            return {"gesture": "none", "confidence": 0.0, "binding": None}
        
        # Get the gesture label
        gesture = id_to_label.get(pred_idx, "unknown")
        
        # Get the binding for this gesture
        binding = binding_map.get(gesture, None)
        
        return {
            "gesture": gesture,
            "confidence": confidence,
            "binding": binding
        }
        
    except Exception as e:
        logging.error(f"Error during gesture recognition: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Gesture recognition failed: {str(e)}"
        )

@router.post("/add_gesture", response_model=TrainingResponse)
async def add_gesture_data(
    background_tasks: BackgroundTasks,
    gesture: str = Form(...),
    binding: str = Form(...),
    user_id: int = Form(...),
    keypoints: str = Form(...),  # JSON string of keypoints
    db: Session = Depends(get_db)
):
    """
    Add new gesture data from a user.
    
    Stores the gesture data in the database and CSV files for training.
    """
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
        
        # Parse keypoints from JSON
        keypoints_data = json.loads(keypoints)
        keypoints_array = np.array(keypoints_data, dtype=np.float32)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Serialize keypoints for database storage
        keypoints_serialized = pickle.dumps(keypoints_array)
        
        # Store in database
        new_gesture = GestureData(
            user_id=user_id,
            label=gesture,
            binding=binding,
            keypoints=keypoints_serialized,
            created_at=datetime.utcnow()
        )
        
        db.add(new_gesture)
        db.commit()
        db.refresh(new_gesture)
        
        # Append to CSV file for training
        gesture_file = f"data/{gesture}.csv"
        
        # Create header if file doesn't exist
        file_exists = os.path.isfile(gesture_file)
        
        # Flatten keypoints array for CSV storage
        flat_keypoints = keypoints_array.flatten()
        
        # Create a DataFrame for the new gesture data
        columns = ["user_id", "label", "binding"] + [f"kp_{i}" for i in range(len(flat_keypoints))]
        
        df = pd.DataFrame([[user_id, gesture, binding] + flat_keypoints.tolist()], columns=columns)
        
        # Append to or create CSV file
        df.to_csv(gesture_file, mode='a', header=not file_exists, index=False)
        
        # Schedule data augmentation and model retraining in the background
        background_tasks.add_task(
            augment_and_retrain, 
            gesture=gesture, 
            binding=binding,
            keypoints_array=keypoints_array
        )
        
        return {
            "success": True,
            "message": "Gesture data added successfully. Model retraining scheduled.",
            "training_started": True
        }
        
    except Exception as e:
        logging.error(f"Error adding gesture data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add gesture data: {str(e)}"
        )

@router.post("/train", response_model=TrainingResponse)
async def trigger_model_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Manually trigger model training.
    
    This will train the model using all available gesture data and 
    update the model files.
    """
    try:
        # Create a new training session record
        training_session = TrainingSession(
            start_time=datetime.utcnow(),
            status="in_progress",
            num_samples=0
        )
        
        db.add(training_session)
        db.commit()
        db.refresh(training_session)
        
        # Schedule model training in the background
        background_tasks.add_task(run_training_task, training_session.id, db)
        
        return {
            "success": True,
            "message": "Model training started in the background",
            "training_started": True
        }
        
    except Exception as e:
        logging.error(f"Error starting model training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model training: {str(e)}"
        )

@router.get("/gestures")
async def get_all_gestures(db: Session = Depends(get_db)):
    """Get a list of all registered gestures with their bindings."""
    try:
        # Get unique gestures and their bindings
        gestures_query = db.query(GestureData.label, GestureData.binding).distinct().all()
        
        gestures = [{"label": g[0], "binding": g[1]} for g in gestures_query]
        
        return {"gestures": gestures}
        
    except Exception as e:
        logging.error(f"Error retrieving gestures: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve gestures: {str(e)}"
        )

@router.get("/training_status/{session_id}")
async def get_training_status(session_id: int, db: Session = Depends(get_db)):
    """Get the status of a specific training session."""
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not training_session:
            raise HTTPException(status_code=404, detail=f"Training session {session_id} not found")
        
        return {
            "id": training_session.id,
            "status": training_session.status,
            "start_time": training_session.start_time,
            "end_time": training_session.end_time,
            "accuracy": training_session.accuracy,
            "error_message": training_session.error_message,
            "num_samples": training_session.num_samples
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logging.error(f"Error retrieving training status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve training status: {str(e)}"
        )

# Background tasks
async def run_training_task(session_id: int, db: Session):
    """Run the model training task in the background."""
    session = next(get_db())
    try:
        # Get the training session
        training_session = session.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not training_session:
            logging.error(f"Training session {session_id} not found")
            return
        
        # Run the training
        result = train_model()
        
        # Update the training session
        training_session.end_time = datetime.utcnow()
        
        if result["success"]:
            training_session.status = "completed"
            training_session.accuracy = result.get("accuracy", 0.0)
            training_session.num_samples = result.get("num_samples", 0)
        else:
            training_session.status = "failed"
            training_session.error_message = result.get("error", "Unknown error")
        
        session.commit()
        
        # Clear the model cache to force reload
        model_cache["model"] = None
        model_cache["metadata"] = None
        
    except Exception as e:
        logging.error(f"Error during training task: {str(e)}")
        if training_session:
            training_session.status = "failed"
            training_session.error_message = str(e)
            training_session.end_time = datetime.utcnow()
            session.commit()
    finally:
        session.close()

async def augment_and_retrain(gesture: str, binding: str, keypoints_array: np.ndarray):
    """Augment gesture data and trigger model retraining."""
    try:
        # Augment the data
        augmented_data = augment_gesture_data(keypoints_array)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Append augmented data to CSV file
        gesture_file = f"data/{gesture}.csv"
        file_exists = os.path.isfile(gesture_file)
        
        # Create columns for the dataframe
        columns = ["user_id", "label", "binding"] + [f"kp_{i}" for i in range(augmented_data.shape[1])]
        
        # Create a dataframe for each augmented sample
        for i, sample in enumerate(augmented_data):
            df = pd.DataFrame([[-1, gesture, binding] + sample.tolist()], columns=columns)
            df.to_csv(gesture_file, mode='a', header=not file_exists and i == 0, index=False)
        
        # Train the model
        session = next(get_db())
        try:
            # Create a new training session record
            training_session = TrainingSession(
                start_time=datetime.utcnow(),
                status="in_progress",
                num_samples=0
            )
            
            session.add(training_session)
            session.commit()
            session.refresh(training_session)
            
            # Run the training task
            await run_training_task(training_session.id, session)
            
        finally:
            session.close()
            
    except Exception as e:
        logging.error(f"Error during augmentation and retraining: {str(e)}")
