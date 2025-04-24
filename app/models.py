"""
Database models for the application.
"""
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, ForeignKey, Boolean, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import os

Base = declarative_base()
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    """User model for storing face authentication data."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    face_encodings = Column(LargeBinary)  # Serialized numpy array of face encodings
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    gestures = relationship("GestureData", back_populates="user")

class GestureData(Base):
    """Model for storing gesture training data."""
    __tablename__ = "gesture_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    label = Column(String, index=True)  # Gesture label (e.g., "swipe_left")
    binding = Column(String)  # Action binding (e.g., "previous_page")
    keypoints = Column(LargeBinary)  # Serialized numpy array of keypoints
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="gestures")

class TrainingSession(Base):
    """Model for tracking model training sessions."""
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String)  # "in_progress", "completed", "failed"
    accuracy = Column(Float)
    error_message = Column(String, nullable=True)
    num_samples = Column(Integer)
    
# Create the database tables
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# Get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
