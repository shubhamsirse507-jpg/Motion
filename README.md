# 👁️ Code Eye — Animated Anime Eye with Motion Tracking

> A real-time webcam-powered, animated anime eye that watches your movements, recognizes hand gestures for math calculations, and listens to your voice — rendered entirely in the browser using HTML5 Canvas.

![Code Eye Banner](https://img.shields.io/badge/Code%20Eye-Motion%20Tracking-blue?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv)
![HTML5](https://img.shields.io/badge/HTML5-Canvas-E34F26?style=for-the-badge&logo=html5&logoColor=white)

---

## ✨ Features

### 👁️ Animated Anime Eye
- Full-screen, beautifully rendered anime-style eye using **HTML5 Canvas**
- Sapphire → Azure → Cyan iris gradient with rotating code-character rings
- Saccadic (snap-smooth) eye movement tracking
- Realistic **blinking** on click, motion events, and random intervals
- Idle wandering animation when no input is detected
- Glints, nebula glow pool, eyelashes, eyebrow, and eyelid crease

### 🎥 Camera PIP (Picture-in-Picture)
- Small live webcam preview panel (bottom-right corner)
- Real-time **MJPEG stream** from Python backend
- Sci-fi HUD overlay with motion coordinates and tracking state
- Minimize / hide / restore controls
- Auto-reconnect fallback when server is offline

### 🧠 Advanced Motion Detection (Python Backend)
- **MOG2 background subtraction** for robust motion detection
- Adaptive velocity vector calculation and **saccadic tracking**
- Non-blocking **threaded camera capture** (zero buffer lag)
- Sub-millisecond JSON state API at 40Hz polling rate

### 🤚 Hand Gesture Math Calculator
- **YCrCb skin segmentation** for hand detection
- Distance-transform palm center detection
- Fingertip recognition (0–5 fingers per hand)
- Dual-hand math: Left Hand **op** Right Hand
  - Add, Subtract, Multiply, Divide
- Live 4-operation breakdown grid
- **Quiz Mode** — solve math problems by holding up fingers!

### 🎤 Voice Recognition (Web Speech API)
- Fully browser-native — no external API needed
- Say a **number** → sets Operand A
- Say **"plus"** / **"minus"** / **"times"** / **"divided"** → selects operator
- Say a second **number** → instantly computes and displays result
- Visual waveform animation while listening
- Optional **Text-to-Speech** readback of results

---

## 🗂️ Project Structure

```
Eye/
├── code_eye.html        # Frontend: Canvas eye, PIP camera, Math panel, Voice UI
├── motion_detector.py   # CV Engine: MOG2, hand detection, gesture math
├── motion_server.py     # HTTP Server: serves HTML, /motion API, /video_feed MJPEG
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A webcam

### 1. Install Dependencies

```bash
pip install opencv-python numpy
```

### 2. Run the Server

```bash
python motion_server.py
```

Optional arguments:

```bash
python motion_server.py --camera 0 --port 8000 --min-area 80
```

| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index (0 = default webcam) |
| `--port` | `8000` | HTTP server port |
| `--min-area` | `80` | Minimum motion area threshold |
| `--history` | `150` | MOG2 background history frames |
| `--var-threshold` | `25` | MOG2 variance threshold |
| `--blink-cooldown` | `0.7` | Minimum seconds between blinks |

### 3. Open in Browser

Navigate to:
```
http://localhost:8000
```

---

## 🎮 Controls & Shortcuts

| Key / Action | Effect |
|---|---|
| **Mouse Move** | Eye follows cursor |
| **Click** | Eye blinks |
| `C` | Toggle camera PIP panel |
| `M` | Toggle Math/Voice panel |
| `+` `-` `*` `/` | Change math operator |
| **Voice: "five"** | Set Operand A = 5 |
| **Voice: "plus"** | Select addition operator |
| **Voice: "three"** | Compute result |

---

## 🧮 Voice Math Flow

```
1. Say a number  ->  "five"      ->  Operand A = 5
2. Say operator  ->  "plus"      ->  Operator = +
3. Say a number  ->  "three"     ->  Result = 8
```

Supported number words: *one, two, three, four, five, six, seven, eight, nine, ten*

Supported operators: *plus, minus, times / multiply, divided / divide*

---

## 🏗️ Architecture

```
Browser (code_eye.html)
    |
    +-- Canvas Eye Renderer (requestAnimationFrame loop)
    +-- Motion Polling  ---- GET /motion (40Hz JSON) ---+
    +-- Camera Stream   ---- GET /video_feed (MJPEG) ---+
    +-- Voice Engine    ---- Web Speech API (local)     |
                                                        |
Python Server (motion_server.py)                        |
    |                                                   |
    +-- ThreadingHTTPServer <---------------------------+
    +-- /motion      -> MotionState JSON
    +-- /video_feed  -> MJPEG multipart stream
    +-- /            -> serves code_eye.html

Motion Detector (motion_detector.py)
    +-- ThreadedCamera          (non-blocking webcam read)
    +-- MOG2 Subtractor         (background modeling)
    +-- Hand Detector           (YCrCb skin + contours)
    +-- AdvancedMotionDetector  -> MotionState + JPEG
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5 Canvas, Vanilla JavaScript, CSS3 |
| Voice | Web Speech API (SpeechRecognition) |
| Backend | Python 3, `http.server.ThreadingHTTPServer` |
| Computer Vision | OpenCV (`cv2`), NumPy |
| Motion Detection | MOG2 Background Subtractor |
| Hand Tracking | YCrCb Skin Segmentation + Convex Hull |

---

## 📄 License

MIT License — feel free to use, modify, and share.

---

> Built with love — *The eye is always watching.*
