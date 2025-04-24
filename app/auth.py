"""
Authentication endpoints for face recognition.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import numpy as np
import pickle
import logging
import time
import base64
import cv2
import os
from datetime import datetime
from typing import List, Optional

from app.models import User, get_db
from app.schemas import UserCreate, AuthResponse, FaceAuthRequest
from app.face_processing import (
    process_face_image, 
    compare_face_encodings, 
    extract_face_encoding
)

router = APIRouter()

@router.post("/register", response_model=AuthResponse)
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Register a new user with face authentication.
    
    Takes a username, email, and face image, processes the face and 
    stores the encodings for future authentication.
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username already registered"
        )
    
    # Process the face image
    try:
        image_bytes = await face_image.read()
        face_encoding = await process_face_image(image_bytes)
        
        if face_encoding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image"
            )
        
        # Serialize the face encodings
        face_encoding_bytes = pickle.dumps(face_encoding)
        
        # Save the face image for the user
        save_path = f"uploads/faces/{username}.jpg"
        with open(save_path, "wb") as f:
            # Reset the file cursor to the beginning and read again
            await face_image.seek(0)
            f.write(await face_image.read())
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            face_encodings=face_encoding_bytes,
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {"message": "User registered successfully", "user_id": new_user.id}
        
    except Exception as e:
        logging.error(f"Error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/authenticate", response_model=AuthResponse)
async def authenticate_user(
    auth_request: FaceAuthRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate a user using face recognition.
    
    Takes a base64-encoded image and performs face comparison with stored encodings.
    """
    try:
        # Decode the base64 image
        image_data = base64.b64decode(auth_request.face_image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image data"
            )
        
        # Extract face encoding from the image
        face_encoding = extract_face_encoding(image)
        
        if face_encoding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image"
            )
        
        # Get user from database
        user = db.query(User).filter(User.username == auth_request.username).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Deserialize stored face encodings
        stored_encoding = pickle.loads(user.face_encodings)
        
        # Compare face encodings
        is_match = compare_face_encodings(face_encoding, stored_encoding)
        
        if is_match:
            # Update last login timestamp
            user.last_login = datetime.utcnow()
            db.commit()
            
            return {"message": "Authentication successful", "user_id": user.id}
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Face authentication failed"
            )
            
    except Exception as e:
        logging.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )

@router.post("/authenticate_frames", response_model=AuthResponse)
async def authenticate_frames(
    auth_request: List[FaceAuthRequest],
    db: Session = Depends(get_db)
):
    """
    Authenticate a user using multiple frames.
    
    Takes a list of base64-encoded images and performs face comparison
    with stored encodings. Designed for periodic authentication during
    video streaming.
    """
    if not auth_request or len(auth_request) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No frames provided"
        )
    
    # Get username from the first frame
    username = auth_request[0].username
    
    # Get user from database
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Deserialize stored face encodings
    stored_encoding = pickle.loads(user.face_encodings)
    
    # Process frames and count matches
    matches = 0
    processed_frames = 0
    
    try:
        for frame_request in auth_request:
            # Decode the base64 image
            image_data = base64.b64decode(frame_request.face_image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Extract face encoding from the image
            face_encoding = extract_face_encoding(image)
            
            if face_encoding is not None:
                processed_frames += 1
                
                # Compare face encodings
                if compare_face_encodings(face_encoding, stored_encoding):
                    matches += 1
        
        # Require at least 30% of frames to have a matching face
        if processed_frames > 0 and matches >= processed_frames * 0.3:
            # Update last login timestamp
            user.last_login = datetime.utcnow()
            db.commit()
            
            return {"message": "Authentication successful", "user_id": user.id}
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Face authentication failed. Matched {matches}/{processed_frames} frames."
            )
            
    except Exception as e:
        logging.error(f"Frame authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )

@router.get("/users")
async def get_users(db: Session = Depends(get_db)):
    """Get a list of all registered users."""
    users = db.query(User).all()
    return [{"id": user.id, "username": user.username, "email": user.email} for user in users]
