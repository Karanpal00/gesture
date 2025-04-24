"""
Initialization module for the gesture recognition and face authentication API.
"""
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging

from app.auth import router as auth_router
from app.gestures import router as gestures_router
from app.config import settings

# Create FastAPI app
app = FastAPI(
    title="Gesture & Face Authentication API",
    description="Backend API for face authentication and gesture recognition",
    version="1.0.0",
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(gestures_router, prefix="/gestures", tags=["Gestures"])

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("uploads/faces", exist_ok=True)
os.makedirs("uploads/gestures", exist_ok=True)

# Root endpoint
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Documentation endpoints are automatically added by FastAPI

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"detail": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return {"detail": "Internal server error", "status_code": 500}
