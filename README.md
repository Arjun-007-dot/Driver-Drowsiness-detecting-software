# Driver-Drowsiness-detectoing-software

A real-time system that detects driver drowsiness using computer vision and machine learning, triggering alerts to prevent fatigue-related accidents.

## Features
- Real-time facial feature detection (eyes, blink rate)
- Micro-sleep detection using machine learning
- Non-intrusive sound/visual alerts
- Configurable sensitivity settings

## Prerequisites
- Python 3.8+
- OpenCV
- Dlib
- Scipy
- Imutils

## Installation
Clone the repository:
   git clone https://github.com/yourusername/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
   
Install dependencies:
pip install -r requirements.txt

Download the dlib shape predictor model and place it in dlib_models/:

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat dlib_models/

Run the application:
python main.py

For building a standalone executable:
pyinstaller --onefile --windowed --add-data "config.json;." --add-data "dlib_models;dlib_models" main.py

Configuration:
Edit config.json to adjust:
EAR (Eye Aspect Ratio) threshold
Alert sensitivity
Frame rate settings

Project Structure
text
├── main.py             # Main application
├── config.json         # Configuration file
├── dlib_models/        # Facial landmark models
├── requirements.txt    # Dependencies
└── README.md           # This file
