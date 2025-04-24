#!/usr/bin/env python
"""
Main entry point for the gesture recognition and face authentication API.
"""
import logging
import os
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__, 
            static_folder="app/static",
            template_folder="app/templates")

# Create required directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/faces", exist_ok=True)
os.makedirs("uploads/gestures", exist_ok=True)

# Root endpoint
@app.route("/")
def root():
    return render_template("index.html")

# API Endpoints
@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})

@app.route("/auth/users", methods=["GET"])
def get_users():
    # Mock data for now
    return jsonify([
        {"id": 1, "username": "test_user", "email": "test@example.com"}
    ])

@app.route("/gestures/gestures", methods=["GET"])
def get_gestures():
    # Mock data for now
    return jsonify({
        "gestures": [
            {"label": "swipe_left", "binding": "previous_page"},
            {"label": "swipe_right", "binding": "next_page"},
            {"label": "hand_up", "binding": "scroll_up"},
            {"label": "hand_down", "binding": "scroll_down"}
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
