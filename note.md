# Problem

# Before (wrong):
ImageFolder was pointed at /content/asl_data/asl_alphabet_train, so it saw only one folder (asl_alphabet_train) and treated it as 1 class.

# After (correct):
ImageFolder is pointed at /content/asl_data/asl_alphabet_train/asl_alphabet_train, where the 29 class folders (A-Z, del, nothing, space) actually exist.

Before: Model learned 1 class → fake 100% accuracy.
After: Model learns 29 classes → real ASL classifier.


ASL Dataset
    ↓
Data Preprocessing
    ↓
EfficientNet-B0 Transfer Learning
    ↓
Training
    ↓
Model Saved (.pth)
    ↓
Webcam Input
    ↓
MediaPipe Hand Detection
    ↓
Hand Crop
    ↓
Model Prediction
    ↓
Letter Output