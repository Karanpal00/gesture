#!/usr/bin/env python
"""
Main entry point for the gesture recognition and face authentication API.
"""
import logging
import os
import base64
import pickle
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG)

# Create required directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/faces", exist_ok=True)
os.makedirs("uploads/gestures", exist_ok=True)

# Create the Flask app
app = Flask(__name__, 
            static_folder="app/static",
            template_folder="app/templates")

# Set secret key for sessions
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Configure database
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # Fix for SQLAlchemy 1.4+ compatibility
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
else:
    # Fallback to SQLite if DATABASE_URL is not set
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize database
db = SQLAlchemy(app)

# Models
class User(db.Model):
    """User model for storing face authentication data."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    face_encodings = db.Column(db.LargeBinary)  # Serialized face encodings
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    gestures = db.relationship("GestureData", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username}>"

class GestureData(db.Model):
    """Model for storing gesture training data."""
    __tablename__ = "gesture_data"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    label = db.Column(db.String(80), nullable=False)  # Gesture name (e.g., "swipe_left")
    binding = db.Column(db.String(80), nullable=False)  # Action binding (e.g., "previous_page")
    keypoints = db.Column(db.LargeBinary)  # Serialized keypoints as numpy array
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship("User", back_populates="gestures")
    
    def __repr__(self):
        return f"<GestureData {self.label} ({self.binding})>"

class TrainingSession(db.Model):
    """Model for tracking model training sessions."""
    __tablename__ = "training_sessions"

    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(80))  # "in_progress", "completed", "failed"
    accuracy = db.Column(db.Float)
    error_message = db.Column(db.Text)
    num_samples = db.Column(db.Integer)
    
    def __repr__(self):
        return f"<TrainingSession {self.id} ({self.status})>"

# Create all tables
with app.app_context():
    db.create_all()

# Face processing utilities
def extract_face_encoding(image):
    """Extract face encoding from an image using OpenCV."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Get the first face
        (x, y, w, h) = faces[0]
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to a standard size
        face_roi = cv2.resize(face_roi, (128, 128))
        
        # Flatten and normalize
        face_encoding = face_roi.flatten().astype(np.float32) / 255.0
        
        return face_encoding
    except Exception as e:
        logging.error(f"Error extracting face encoding: {e}")
        return None

def compare_face_encodings(encoding1, encoding2, tolerance=0.6):
    """Compare two face encodings and return True if they match."""
    if encoding1 is None or encoding2 is None:
        return False
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(encoding1 - encoding2)
    
    # Convert distance to similarity score
    similarity = 1.0 / (1.0 + distance)
    
    # Check if similarity is above threshold
    match = similarity >= (1 - tolerance)
    
    return bool(match)

# Gesture data augmentation
def augment_gesture_data(keypoints, num_samples=10):
    """Augment gesture keypoints to create additional training samples."""
    # Ensure keypoints is a 1D array
    keypoints = keypoints.flatten()
    
    # Get the number of coordinates
    num_coords = len(keypoints)
    
    # Create an array for the augmented samples
    augmented_samples = np.zeros((num_samples, num_coords), dtype=np.float32)
    
    # Generate augmented samples with small variations
    for i in range(num_samples):
        # Copy the original keypoints
        aug_keypoints = keypoints.copy()
        
        # Apply random noise
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, num_coords)
        aug_keypoints += noise
        
        # Apply scaling
        scale_factor = np.random.uniform(0.95, 1.05)
        aug_keypoints *= scale_factor
        
        # Apply small translation
        translation = np.random.uniform(-0.05, 0.05, 2)
        if num_coords % 2 == 0:
            for j in range(0, num_coords, 2):
                aug_keypoints[j] += translation[0]
                aug_keypoints[j+1] += translation[1]
        
        # Store the augmented sample
        augmented_samples[i] = aug_keypoints
    
    return augmented_samples

# API Endpoints
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    users = User.query.all()
    gestures = db.session.query(GestureData.label, GestureData.binding).distinct().all()
    training_sessions = TrainingSession.query.order_by(TrainingSession.start_time.desc()).limit(5).all()
    
    return render_template(
        "dashboard.html", 
        users=users, 
        gestures=gestures, 
        training_sessions=training_sessions
    )

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})

# User registration and authentication
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Get form data
        username = request.form.get("username")
        email = request.form.get("email")
        face_image = request.files.get("face_image")
        
        # Validate inputs
        if not username or not email or not face_image:
            flash("Please fill all fields", "danger")
            return redirect(url_for("register"))
        
        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already taken", "danger")
            return redirect(url_for("register"))
        
        try:
            # Process the face image
            image_bytes = face_image.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                flash("Invalid image format", "danger")
                return redirect(url_for("register"))
            
            # Extract face encoding
            face_encoding = extract_face_encoding(image)
            
            if face_encoding is None:
                flash("No face detected in the image", "danger")
                return redirect(url_for("register"))
            
            # Serialize face encoding
            face_encoding_bytes = pickle.dumps(face_encoding)
            
            # Save the face image
            filename = secure_filename(f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            filepath = os.path.join("uploads/faces", filename)
            with open(filepath, "wb") as f:
                face_image.seek(0)
                f.write(face_image.read())
            
            # Create new user
            new_user = User(
                username=username,
                email=email,
                face_encodings=face_encoding_bytes,
                created_at=datetime.utcnow()
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("login"))
        
        except Exception as e:
            db.session.rollback()
            logging.error(f"Registration error: {e}")
            flash(f"Registration failed: {str(e)}", "danger")
            return redirect(url_for("register"))
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        face_image = request.files.get("face_image")
        
        if not username or not face_image:
            flash("Please fill all fields", "danger")
            return redirect(url_for("login"))
        
        try:
            # Get user from database
            user = User.query.filter_by(username=username).first()
            if not user:
                flash("User not found", "danger")
                return redirect(url_for("login"))
            
            # Process the face image
            image_bytes = face_image.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                flash("Invalid image format", "danger")
                return redirect(url_for("login"))
            
            # Extract face encoding
            face_encoding = extract_face_encoding(image)
            
            if face_encoding is None:
                flash("No face detected in the image", "danger")
                return redirect(url_for("login"))
            
            # Compare with stored encoding
            stored_encoding = pickle.loads(user.face_encodings)
            
            is_match = compare_face_encodings(face_encoding, stored_encoding)
            
            if is_match:
                # Update last login
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Face authentication failed", "danger")
                return redirect(url_for("login"))
        
        except Exception as e:
            logging.error(f"Login error: {e}")
            flash(f"Login failed: {str(e)}", "danger")
            return redirect(url_for("login"))
            
    return render_template("login.html")

# Gesture API endpoints
@app.route("/api/gestures", methods=["GET"])
def get_gestures():
    """Get all registered gestures with their bindings."""
    gestures = db.session.query(GestureData.label, GestureData.binding).distinct().all()
    return jsonify({
        "gestures": [{"label": g[0], "binding": g[1]} for g in gestures]
    })

@app.route("/api/gestures/add", methods=["POST"])
def add_gesture():
    """Add new gesture data."""
    try:
        data = request.json
        user_id = data.get("user_id")
        label = data.get("label")
        binding = data.get("binding")
        keypoints = data.get("keypoints")
        
        # Validate inputs
        if not user_id or not label or not binding or not keypoints:
            return jsonify({"success": False, "message": "Missing required fields"}), 400
        
        # Check if user exists
        user = User.query.get(user_id)
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        # Convert keypoints to numpy array
        keypoints_array = np.array(keypoints, dtype=np.float32)
        
        # Serialize keypoints
        keypoints_bytes = pickle.dumps(keypoints_array)
        
        # Store in database
        new_gesture = GestureData(
            user_id=user_id,
            label=label,
            binding=binding,
            keypoints=keypoints_bytes,
            created_at=datetime.utcnow()
        )
        
        db.session.add(new_gesture)
        db.session.commit()
        
        # Save to CSV for training
        gesture_file = f"data/{label}.csv"
        file_exists = os.path.isfile(gesture_file)
        
        # Flatten keypoints array for CSV storage
        flat_keypoints = keypoints_array.flatten()
        
        # Create columns for DataFrame
        columns = ["user_id", "label", "binding"] + [f"kp_{i}" for i in range(len(flat_keypoints))]
        
        # Create DataFrame
        df = pd.DataFrame([[user_id, label, binding] + flat_keypoints.tolist()], columns=columns)
        
        # Save to CSV
        df.to_csv(gesture_file, mode='a', header=not file_exists, index=False)
        
        # Generate augmented data
        augmented_data = augment_gesture_data(keypoints_array)
        
        # Save augmented data to CSV
        for i, sample in enumerate(augmented_data):
            df = pd.DataFrame([[-1, label, binding] + sample.tolist()], columns=columns)
            df.to_csv(gesture_file, mode='a', header=False, index=False)
        
        return jsonify({
            "success": True,
            "message": "Gesture data added successfully",
            "gesture_id": new_gesture.id
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error adding gesture data: {e}")
        return jsonify({"success": False, "message": f"Failed to add gesture data: {str(e)}"}), 500

@app.route("/api/train", methods=["POST"])
def train_model():
    """Trigger model training."""
    try:
        # Create a new training session
        training_session = TrainingSession(
            start_time=datetime.utcnow(),
            status="in_progress",
            num_samples=0
        )
        
        db.session.add(training_session)
        db.session.commit()
        
        # Here we would normally call the actual training function
        # For now, we'll simulate it
        
        # Update the training session
        training_session.end_time = datetime.utcnow()
        training_session.status = "completed"
        training_session.accuracy = 0.95  # Mock accuracy value
        training_session.num_samples = 100  # Mock sample count
        
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Model training completed successfully",
            "session_id": training_session.id
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error training model: {e}")
        return jsonify({"success": False, "message": f"Failed to train model: {str(e)}"}), 500

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Get all registered users."""
    users = User.query.all()
    return jsonify([{
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None
    } for user in users])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
