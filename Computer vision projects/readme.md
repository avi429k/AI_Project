# 🎭 Face Detection System

A real-time face detection application built with Python and OpenCV, using a pre-trained Haar Cascade classifier to detect and highlight faces via webcam feed.

---

## 📸 Demo

> The system draws bounding boxes around detected faces in real time and displays a live face count on screen.

---

## 🚀 Features

- Real-time face detection via webcam
- Haar Cascade classifier for fast and reliable detection
- Live face count overlay on video feed
- Save captured frames as images with a single keypress
- Lightweight and easy to run locally

<img width="1135" height="712" alt="image" src="https://github.com/user-attachments/assets/ecb2699e-e415-4cfb-a3dd-d30437639b5d" />

---

## 🛠️ Tech Stack

- *Language:* Python 3.x
- *Library:* [OpenCV](https://opencv.org/) (cv2)
- *Model:* Haar Cascade — haarcascade_frontalface_default.xml (bundled with OpenCV)

---

## 📦 Installation

1. *Clone the repository*

bash
git clone https://github.com/your-username/face-detection.git
cd face-detection


2. *Create a virtual environment (optional but recommended)*

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. *Install dependencies*

bash
pip install opencv-python


---

## ▶️ Usage

bash
python face_detection.py


### Controls

| Key | Action |
|-----|--------|
| s | Save the current frame as captured_face.jpg |
| q | Quit the application |

---

## ⚙️ Configuration

You can tweak the detection parameters in the script to adjust sensitivity:

python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,   # How much the image is scaled down at each step
    minNeighbors=5,    # Higher = fewer false positives
)


If your webcam isn't detected, change the camera index:

python
cap = cv2.VideoCapture(0)  # Try 0, 1, or 2 depending on your system


> *macOS users:* The script uses cv2.CAP_AVFOUNDATION as the backend, which is the recommended capture API for macOS.

---

## 📁 Project Structure


face-detection/
│
├── face_detection.py       # Main script
├── captured_face.jpg       # Saved on keypress (auto-generated)
└── README.md


---

## 🧠 How It Works

1. Loads OpenCV's pre-trained Haar Cascade frontal face model
2. Captures live video from the webcam frame by frame
3. Converts each frame to grayscale (required for Haar detection)
4. Runs detectMultiScale to find face regions
5. Draws green rectangles around detected faces and shows the count
6. Displays the annotated frame in a window in real time

---

## 📋 Requirements

- Python 3.7+
- OpenCV 4.x (opencv-python)
- A connected webcam

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [OpenCV](https://opencv.org/) for the computer vision library and pre-trained Haar Cascade models
