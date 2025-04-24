# Gesture Data Directory

This directory contains CSV files for gesture training data. Each gesture is stored in a separate CSV file named after the gesture (e.g., `swipe_left.csv`).

## File Format

Each CSV file has the following columns:
- `user_id`: ID of the user who recorded the gesture
- `label`: Gesture label (e.g., "swipe_left")
- `binding`: Action binding (e.g., "previous_page")
- `kp_0`, `kp_1`, ...: Hand keypoint coordinates

## Adding New Gestures

New gesture data is automatically saved here when:
1. Users add gestures through the API
2. Augmented gesture data is generated for training

## Data Privacy

Be mindful that this directory contains user-contributed training data. Handle this data according to relevant privacy regulations and policies.
