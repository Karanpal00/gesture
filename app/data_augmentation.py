"""
Data augmentation functions for gesture keypoints.
"""
import numpy as np
import logging
from typing import List
from app.config import settings

def augment_gesture_data(keypoints: np.ndarray, num_samples: int = None) -> np.ndarray:
    """
    Augment gesture keypoints data.
    
    Applies various transformations to the keypoints to create 
    additional training samples.
    
    Args:
        keypoints: Original keypoints as a numpy array
        num_samples: Number of augmented samples to generate
                     (defaults to settings.AUGMENTATION_SAMPLES)
                     
    Returns:
        Array of augmented samples
    """
    if num_samples is None:
        num_samples = settings.AUGMENTATION_SAMPLES
    
    # Ensure keypoints is a 1D array
    keypoints = keypoints.flatten()
    
    # Get the number of keypoints (x, y coordinates)
    num_coords = len(keypoints)
    
    # Create an empty array for augmented samples
    augmented_samples = np.zeros((num_samples, num_coords), dtype=np.float32)
    
    # Generate augmented samples
    for i in range(num_samples):
        # Create a copy of the original keypoints
        aug_keypoints = keypoints.copy()
        
        # Apply random noise (small jitter)
        noise_scale = 0.01  # Scale of the noise
        noise = np.random.normal(0, noise_scale, num_coords)
        aug_keypoints += noise
        
        # Apply scaling (make slightly larger or smaller)
        scale_factor = np.random.uniform(0.95, 1.05)
        aug_keypoints *= scale_factor
        
        # Apply rotation (only for 2D keypoints organized as [x1, y1, x2, y2, ...])
        if num_coords % 2 == 0:  # Make sure we have an even number of coordinates
            # Reshape to (num_points, 2) to make rotation easier
            points = aug_keypoints.reshape(-1, 2)
            
            # Generate a random rotation angle (in radians)
            angle = np.random.uniform(-0.1, 0.1)  # Small rotation
            
            # Create rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Apply rotation
            rotated_points = np.dot(points, rotation_matrix)
            
            # Flatten back
            aug_keypoints = rotated_points.flatten()
        
        # Add translation (shift keypoints)
        translation = np.random.uniform(-0.05, 0.05, 2)  # x, y translation
        if num_coords % 2 == 0:
            for j in range(0, num_coords, 2):
                aug_keypoints[j] += translation[0]
                aug_keypoints[j+1] += translation[1]
        
        # Store the augmented sample
        augmented_samples[i] = aug_keypoints
    
    logging.info(f"Generated {num_samples} augmented samples for gesture data")
    return augmented_samples

def augment_face_data(face_encoding: np.ndarray, num_samples: int = 5) -> List[np.ndarray]:
    """
    Augment face encoding data.
    
    This function is more conservative with face data since face encodings
    are sensitive to changes.
    
    Args:
        face_encoding: Original face encoding
        num_samples: Number of augmented samples to generate
        
    Returns:
        List of augmented face encodings
    """
    augmented_samples = []
    
    # Get the number of dimensions
    num_dims = len(face_encoding)
    
    # Generate augmented samples
    for _ in range(num_samples):
        # Create a copy of the original encoding
        aug_encoding = face_encoding.copy()
        
        # Apply very small random noise
        noise_scale = 0.001  # Keep this value very small
        noise = np.random.normal(0, noise_scale, num_dims)
        aug_encoding += noise
        
        # Normalize the encoding to unit length (important for face_recognition)
        aug_encoding = aug_encoding / np.linalg.norm(aug_encoding)
        
        augmented_samples.append(aug_encoding)
    
    logging.info(f"Generated {num_samples} augmented samples for face data")
    return augmented_samples
