# Driver Drowsiness Detection System

A **real-time driver drowsiness detection** system using **computer vision** and **machine learning** to monitor driver alertness. It helps prevent fatigue-related accidents by issuing alerts when signs of drowsiness are detected.

---

## 🚘 Features

- 🔍 Real-time **facial feature tracking** (eye aspect ratio, blink rate)
- 🧠 **Micro-sleep detection** powered by machine learning
- 🔔 **Non-intrusive alerts** (audio & visual) to wake drowsy drivers
- ⚙️ **Customizable sensitivity** and configuration via `config.json`

---

## 🛠️ Prerequisites

- Python **3.8+**
- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)
- Scipy  
- Imutils

---

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dlib facial landmark model:**

   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   mv shape_predictor_68_face_landmarks.dat dlib_models/
   ```

---

## 🚀 Running the Application

```bash
python main.py
```

To create a standalone executable (Windows/Linux):

```bash
pyinstaller --onefile --windowed \
--add-data "config.json;." \
--add-data "dlib_models;dlib_models" main.py
```

---

## ⚙️ Configuration

Edit `config.json` to customize the detection behavior:

```json
{
  "EAR_THRESHOLD": 0.25,
  "CONSEC_FRAMES": 20,
  "ALERT_VOLUME": 0.8,
  "FRAME_RATE": 24
}
```

- `EAR_THRESHOLD`: Eye aspect ratio threshold for drowsiness
- `CONSEC_FRAMES`: Number of consecutive frames with low EAR to trigger an alert
- `ALERT_VOLUME`: Volume of alert sound
- `FRAME_RATE`: Camera frame processing rate

---

## 📁 Project Structure

```
driver-drowsiness-detection/
├── main.py             # Main application logic
├── config.json         # Detection parameters
├── dlib_models/        # Dlib landmark model
│   └── shape_predictor_68_face_landmarks.dat
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

