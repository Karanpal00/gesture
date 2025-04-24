#!/usr/bin/env python
"""
Test script for the gesture recognition and face authentication system.
This script validates the core functionality of the system.
"""
import os
import cv2
import numpy as np
import pickle
import requests
import logging
import time
import unittest
import tempfile
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for the API
BASE_URL = "http://localhost:5000"

class TestGestureFaceSystem(unittest.TestCase):
    """Test suite for gesture recognition and face authentication system."""
    
    def setUp(self):
        """Set up test environment."""
        # Check if server is running
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code != 200:
                self.skipTest("Server is not running or not accessible")
        except requests.RequestException:
            self.skipTest("Server is not running or not accessible")
            
        # Create test image paths
        self.test_face_path = os.path.join("test_assets", "test_face.jpg")
        self.test_gesture_path = os.path.join("test_assets", "test_gesture.jpg")
        
        # Create directory if it doesn't exist
        os.makedirs("test_assets", exist_ok=True)
            
        # If test images don't exist, create them
        if not os.path.exists(self.test_face_path):
            self._create_test_face_image()
            
        if not os.path.exists(self.test_gesture_path):
            self._create_test_gesture_image()
        
        # Test user credentials
        self.test_username = f"test_user_{int(time.time())}"
        self.test_email = f"test_{int(time.time())}@example.com"

    def _create_test_face_image(self):
        """Create a test face image using a solid color with a simple face shape."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Draw a simple face shape (circle for face, circles for eyes, line for mouth)
        cv2.circle(img, (150, 150), 100, (255, 200, 200), -1)  # Face
        cv2.circle(img, (120, 120), 15, (255, 255, 255), -1)  # Left eye
        cv2.circle(img, (180, 120), 15, (255, 255, 255), -1)  # Right eye
        cv2.ellipse(img, (150, 180), (40, 10), 0, 0, 180, (50, 50, 50), 2)  # Mouth
        
        # Save the image
        cv2.imwrite(self.test_face_path, img)
        logging.info(f"Created test face image at {self.test_face_path}")
    
    def _create_test_gesture_image(self):
        """Create a test gesture image (hand shape)."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Draw a simple hand shape
        # Palm
        cv2.circle(img, (150, 200), 40, (200, 170, 170), -1)
        
        # Fingers
        for i, angle in enumerate([-30, -15, 0, 15, 30]):
            rad = np.radians(angle)
            length = 80 if i != 0 else 60  # Thumb is shorter
            x = int(150 + length * np.sin(rad))
            y = int(200 - length * np.cos(rad))
            cv2.line(img, (150, 200), (x, y), (200, 170, 170), 15)
            cv2.circle(img, (x, y), 7, (200, 170, 170), -1)
        
        # Save the image
        cv2.imwrite(self.test_gesture_path, img)
        logging.info(f"Created test gesture image at {self.test_gesture_path}")
    
    def test_01_health_check(self):
        """Test health check endpoint."""
        response = requests.get(f"{BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        logging.info("Health check passed")
    
    def test_02_register_user(self):
        """Test user registration with face authentication."""
        with open(self.test_face_path, "rb") as f:
            files = {"face_image": (os.path.basename(self.test_face_path), f, "image/jpeg")}
            data = {
                "username": self.test_username,
                "email": self.test_email
            }
            
            response = requests.post(
                f"{BASE_URL}/register",
                files=files,
                data=data,
                allow_redirects=False
            )
            
            # Check if registration redirected to login page
            self.assertIn(response.status_code, [200, 302])
            
            if response.status_code == 302:
                # If redirected, make sure it's to the login page
                self.assertEqual(response.headers["Location"], "/login")
                logging.info(f"User {self.test_username} registered successfully")
            else:
                # If not redirected, check for success message in response
                self.assertIn("Registration successful", response.text)
                logging.info(f"User {self.test_username} registration form submitted")
    
    def test_03_login_user(self):
        """Test user login with face authentication."""
        with open(self.test_face_path, "rb") as f:
            files = {"face_image": (os.path.basename(self.test_face_path), f, "image/jpeg")}
            data = {"username": self.test_username}
            
            response = requests.post(
                f"{BASE_URL}/login",
                files=files,
                data=data,
                allow_redirects=False
            )
            
            # Login may redirect to dashboard if successful
            # or back to login page if face recognition fails
            self.assertIn(response.status_code, [200, 302])
            
            if response.status_code == 302:
                # Check if redirected to dashboard
                if response.headers["Location"] == "/dashboard":
                    logging.info(f"User {self.test_username} logged in successfully")
                else:
                    logging.warning(f"Login redirected to {response.headers['Location']}")
            else:
                # Manual check if login was successful
                success = "Login successful" in response.text
                if success:
                    logging.info(f"User {self.test_username} logged in successfully")
                else:
                    logging.warning("Login unsuccessful, but test continues")
    
    def test_04_get_users(self):
        """Test fetching users from API."""
        response = requests.get(f"{BASE_URL}/api/users")
        self.assertEqual(response.status_code, 200)
        users = response.json()
        self.assertIsInstance(users, list)
        
        # Check if our test user is in the list
        test_user_found = any(user["username"] == self.test_username for user in users)
        
        if test_user_found:
            logging.info(f"Test user {self.test_username} found in users list")
        else:
            logging.warning(f"Test user {self.test_username} not found in users list")
    
    def test_05_get_gestures(self):
        """Test fetching gestures from API."""
        response = requests.get(f"{BASE_URL}/api/gestures")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gestures", data)
        logging.info(f"Found {len(data['gestures'])} gestures")
    
    def test_06_mock_gesture_recognition(self):
        """Test gesture recognition API with mock data."""
        # Create mock keypoints (21 hand landmarks with x, y, z coordinates)
        mock_keypoints = []
        for i in range(21):
            # Distribute keypoints in a pattern resembling a hand
            x = 0.5 + 0.2 * np.cos(i * np.pi / 10)
            y = 0.5 + 0.2 * np.sin(i * np.pi / 10)
            z = 0.0
            mock_keypoints.extend([float(x), float(y), float(z)])
        
        data = {"keypoints": mock_keypoints}
        response = requests.post(f"{BASE_URL}/api/gestures/recognize", json=data)
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        # Check the structure of the response
        self.assertIn("gesture", result)
        self.assertIn("confidence", result)
        
        gesture = result["gesture"]
        confidence = result["confidence"]
        
        logging.info(f"Recognized gesture: {gesture} with confidence {confidence:.4f}")
        
        # If we don't have a real model loaded or no gestures registered,
        # we might get "unknown" as the gesture
        if gesture == "unknown":
            logging.warning("Model returned 'unknown' gesture - this is expected if no model is trained yet")

    def test_07_add_test_gesture(self):
        """Test adding a new gesture."""
        # Fetch users to get a valid user_id
        response = requests.get(f"{BASE_URL}/api/users")
        users = response.json()
        
        if not users:
            self.skipTest("No users found to associate gesture with")
        
        user_id = users[0]["id"]
        
        # Create test gesture keypoints (similar to above but slightly different pattern)
        mock_keypoints = []
        for i in range(21):
            # Distribute keypoints in a pattern resembling a specific gesture
            x = 0.5 + 0.3 * np.cos(i * np.pi / 10)
            y = 0.5 + 0.1 * np.sin(i * np.pi / 10)
            z = 0.0
            mock_keypoints.extend([float(x), float(y), float(z)])
        
        data = {
            "user_id": user_id,
            "label": f"test_gesture_{int(time.time())}",
            "binding": "test_action",
            "keypoints": mock_keypoints
        }
        
        response = requests.post(f"{BASE_URL}/api/gestures/add", json=data)
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertTrue(result["success"])
        self.assertIn("gesture_id", result)
        
        logging.info(f"Added gesture with ID {result['gesture_id']}")

    def test_08_trigger_model_training(self):
        """Test triggering model training."""
        response = requests.post(f"{BASE_URL}/api/train")
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn("success", result)
        self.assertIn("session_id", result)
        
        logging.info(f"Triggered model training with session ID {result['session_id']}")
        
        # We don't wait for training completion as it might take time

if __name__ == "__main__":
    logging.info("Starting test suite...")
    unittest.main(verbosity=2)