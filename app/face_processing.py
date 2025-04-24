"""
Face processing utilities for authentication.
"""
import numpy as np
import cv2
import logging
import hashlib
from typing import Optional, List, Tuple

def extract_face_encoding(image, model: str = "hog") -> Optional[np.ndarray]:
    """
    Extract face encoding from an image using OpenCV.
    
    Args:
        image: Image as a numpy array (BGR format from OpenCV)
        model: Model parameter (kept for compatibility, not used)
        
    Returns:
        Face encoding as numpy array or None if no face is detected
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's pre-trained face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Use the first face detected
    (x, y, w, h) = faces[0]
    
    # Extract face region
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to a standard size to ensure consistent feature vectors
    face_roi = cv2.resize(face_roi, (128, 128))
    
    # Flatten and normalize
    face_encoding = face_roi.flatten().astype(np.float32) / 255.0
    
    return face_encoding

async def process_face_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Process an uploaded face image and extract facial encodings.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Face encoding as numpy array or None if no face is detected
    """
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logging.error("Failed to decode image")
            return None
            
        # Extract face encoding
        face_encoding = extract_face_encoding(image)
        
        if face_encoding is None:
            logging.warning("No face detected in the image")
            
        return face_encoding
        
    except Exception as e:
        logging.error(f"Error processing face image: {str(e)}")
        return None

def compare_face_encodings(encoding1: np.ndarray, encoding2: np.ndarray, tolerance: float = 0.6) -> bool:
    """
    Compare two face encodings to determine if they are the same person.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        tolerance: Tolerance for face comparison (lower is stricter)
        
    Returns:
        True if faces match, False otherwise
    """
    if encoding1 is None or encoding2 is None:
        return False
    
    # Calculate Euclidean distance between the encodings
    distance = np.linalg.norm(encoding1 - encoding2)
    
    # Convert distance to a similarity score (inverse relationship)
    similarity = 1.0 / (1.0 + distance)
    
    # Check if similarity is above threshold (1 - tolerance)
    match = similarity >= (1 - tolerance)
    
    return bool(match)

def get_face_location(image) -> Tuple[List[Tuple[int, int, int, int]], List]:
    """
    Get face location and landmarks from an image.
    
    Args:
        image: Image as a numpy array (BGR format from OpenCV)
        
    Returns:
        Tuple of (face_locations, face_landmarks)
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Convert to format similar to face_recognition library
    face_locations = []
    for (x, y, w, h) in faces:
        # Format as (top, right, bottom, left)
        face_locations.append((y, x+w, y+h, x))
    
    # We don't have facial landmarks with this method, so return empty list
    face_landmarks = []
    
    return face_locations, face_landmarks
