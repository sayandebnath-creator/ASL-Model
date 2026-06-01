# ASL Sign Language Translator

A real-time American Sign Language (ASL) recognition system that uses computer vision and deep learning to identify hand gestures from a webcam feed.

The project combines MediaPipe hand tracking with an EfficientNet-B0 classifier trained on the ASL Alphabet dataset to recognize ASL signs and display predictions in real time.

## Features

* Real-time webcam inference
* MediaPipe-based hand detection and tracking
* EfficientNet-B0 transfer learning model
* Dynamic hand bounding boxes
* Confidence score visualization
* Automatic hand cropping before classification
* Supports ASL alphabet recognition
* CPU and GPU compatible

## Tech Stack

* Python
* PyTorch
* TorchVision
* OpenCV
* MediaPipe
* EfficientNet-B0
* NumPy
* Pillow

## Project Pipeline

```text
Webcam Feed
      ↓
MediaPipe Hand Detection
      ↓
Hand Bounding Box Extraction
      ↓
Image Preprocessing
      ↓
EfficientNet-B0 Classifier
      ↓
ASL Letter Prediction
      ↓
Confidence Score Display
```

## Dataset

The model was trained on the ASL Alphabet Dataset containing approximately 87,000 images across 29 classes.

Classes include:

* A-Z
* space
* delete
* nothing

## Installation

Clone the repository:

```bash
git clone https://github.com/sayandebnath-creator/ASL-Model.git
cd ASL-Model
```

Create a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Place the trained model file in the project root:

```text
best_asl_model.pth
```

Run:

```bash
python webcam.py
```

## Results

* Achieved high classification accuracy using transfer learning
* Real-time inference on webcam feed
* Dynamic hand tracking using MediaPipe
* Low-latency predictions on consumer hardware

## Future Improvements

* Prediction smoothing
* Word formation from letters
* Text-to-speech output
* Sentence-level recognition
* Indian Sign Language (ISL) support
* Web deployment with Gradio
* Mobile application support
