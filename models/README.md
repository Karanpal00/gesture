# Model Files Directory

This directory contains the trained models and metadata for gesture recognition.

## Files

- `gesture_clf_pt.pt`: PyTorch model file
- `gesture_clf_pt.onnx`: ONNX model file (used for inference)
- `meta_pt.pkl`: Metadata pickle file containing:
  - `scaler`: StandardScaler for normalizing input features
  - `label_map`: Mapping from gesture names to numerical indices
  - `binding_map`: Mapping from gesture names to action bindings

## Model Training

The models are automatically trained or retrained when:
- The API endpoint `/gestures/train` is called
- New gesture data is added and background retraining is triggered

## Model Architecture

The model is a simple MLP (Multi-Layer Perceptron) with the following architecture:
- Input layer: Size matches the number of keypoint features
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: Size matches the number of gesture classes

## Using the Models

The ONNX model is used by the API for gesture recognition inference. The metadata file is required for preprocessing input data and interpreting the model's outputs.
