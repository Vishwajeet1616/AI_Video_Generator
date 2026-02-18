# 🎮 AI Gameplay Highlight Generator

An AI-powered web application that automatically generates gameplay highlight reels using audio-based machine learning detection.

---

## 🚀 Overview

This project uses a pretrained audio classification model (YAMNet from TensorFlow Hub) to detect:

- Gunshots
- Explosions
- Machine gun fire
- Laughter
- Screaming
- Cheering

Detected segments are automatically merged and exported as a highlight video.

---

## 🧠 How It Works

1. User uploads gameplay video
2. Audio is extracted at 16kHz mono
3. YAMNet model performs sound classification
4. High-confidence events are detected
5. Overlapping timestamps are merged
6. Final highlight reel is generated

---

## 🛠 Tech Stack

- Python
- Flask
- TensorFlow
- TensorFlow Hub
- MoviePy
- HTML & CSS

---

## ⚙ Installation

1. Clone repository

2. Install dependencies:

pip install -r requirements.txt

3. Run:

python app.py

4. Open in browser:

http://127.0.0.1:5000/

---

## 🎯 Features

- Automated highlight detection
- End-to-end ML pipeline
- Web-based upload interface
- Timestamp merging logic
- Automated video export

---

## 🔥 Future Improvements

- Add computer vision-based kill detection
- Add adjustable confidence threshold
- Add cloud deployment
- Optimize long video processing

---

## 👨‍💻 Author

Vishwajeet Pawar  
AI & Machine Learning Enthusiast
