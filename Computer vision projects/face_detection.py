# ============================================================
# Real-Time Face Detection Project using OpenCV
# Author: Avik Mondal
# Features: Face Detection, Emotion Estimation, Blink Detection,
#            Age/Gender Estimation, Face Count, Save Image
# ============================================================

import cv2
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# Load Haar Cascade Models
# ─────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

# ─────────────────────────────────────────────
# Start Webcam (Windows compatible)
# ─────────────────────────────────────────────
# Try camera index 0, 1, 2 until one works
cap = None
for index in range(3):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows DirectShow
    if cap.isOpened():
        print(f"✅ Webcam opened successfully at index {index}")
        break

if cap is None or not cap.isOpened():
    print("❌ Error: Could not open any webcam.")
    print("→ Make sure your webcam is connected and not used by another app.")
    exit()

# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=" * 45)
print("   🎯 Real-Time Face Detection System")
print("=" * 45)
print("  Press 'S'  → Save current frame")
print("  Press 'Q'  → Quit")
print("  Press 'B'  → Toggle blur background")
print("=" * 45)

# ─────────────────────────────────────────────
# State Variables
# ─────────────────────────────────────────────
blur_bg        = False
saved_count    = 0
frame_count    = 0
blink_counters = {}   # track blinks per face ID (approximate)

# ─────────────────────────────────────────────
# Helper: Estimate Emotion (rule-based AI)
# ─────────────────────────────────────────────
def estimate_emotion(face_roi_gray):
    """
    Simple rule-based emotion estimator using smile detection.
    Returns a label string.
    """
    smiles = smile_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.7,
        minNeighbors=22,
        minSize=(25, 25)
    )
    if len(smiles) > 0:
        return "😊 Happy", (0, 255, 100)
    return "😐 Neutral", (200, 200, 200)

# ─────────────────────────────────────────────
# Helper: Estimate Age Group (rule-based)
# ─────────────────────────────────────────────
def estimate_age_group(face_w):
    """
    Very rough estimate based on face size in frame.
    For accurate results, use a DNN model.
    """
    if face_w < 80:
        return "Child?"
    elif face_w < 130:
        return "Teen/Adult"
    else:
        return "Adult"

# ─────────────────────────────────────────────
# Helper: Draw Rounded Rectangle
# ─────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img,  (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img,  (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img,  (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img,  (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0,   0, 90,  color, thickness)

# ─────────────────────────────────────────────
# Helper: Draw HUD overlay
# ─────────────────────────────────────────────
def draw_hud(frame, face_count, saved_count, blur_bg):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"Faces: {face_count}",
                (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 180), 2)
    cv2.putText(frame, f"Saved: {saved_count}",
                (180, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)
    cv2.putText(frame, timestamp,
                (w - 290, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Bottom hint bar
    hint = "S=Save  Q=Quit  B=Blur BG"
    if blur_bg:
        hint += "  [BLUR ON]"
    cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, hint,
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Check webcam connection.")
        break

    frame_count += 1
    display = frame.copy()

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improve detection in low light

    # ── Detect Faces ──────────────────────────
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    # ── Blur background (AI feature) ──────────
    if blur_bg and len(faces) > 0:
        blurred = cv2.GaussianBlur(display, (55, 55), 0)
        mask = np.zeros(display.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in faces:
            cv2.ellipse(mask,
                        (x + w // 2, y + h // 2),
                        (w // 2 + 20, h // 2 + 30),
                        0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_3ch = cv2.merge([mask, mask, mask])
        display = np.where(mask_3ch > 127, display, blurred)

    # ── Process Each Face ─────────────────────
    for i, (x, y, w, h) in enumerate(faces):

        face_gray = gray[y:y+h, x:x+w]
        face_color = display[y:y+h, x:x+w]

        # Emotion
        emotion_label, emotion_color = estimate_emotion(face_gray)

        # Age group
        age_label = estimate_age_group(w)

        # Eyes (blink detection approximation)
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )
        eye_status = f"Eyes: {len(eyes)}"

        # ── Draw rounded box around face ──────
        draw_rounded_rect(display, (x, y), (x + w, y + h), (0, 200, 255), 2)

        # ── Face label background ─────────────
        label_bg_y = y - 70 if y > 80 else y + h + 5
        cv2.rectangle(display,
                      (x, label_bg_y),
                      (x + w, label_bg_y + 65),
                      (20, 20, 20), -1)
        cv2.rectangle(display,
                      (x, label_bg_y),
                      (x + w, label_bg_y + 65),
                      (0, 200, 255), 1)

        # ── Labels ───────────────────────────
        cv2.putText(display, f"Face #{i+1}  {age_label}",
                    (x + 5, label_bg_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        cv2.putText(display, emotion_label,
                    (x + 5, label_bg_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1)
        cv2.putText(display, eye_status,
                    (x + 5, label_bg_y + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

        # ── Draw eye dots ─────────────────────
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(display,
                       (x + ex + ew // 2, y + ey + eh // 2),
                       4, (0, 255, 0), -1)

    # ── HUD overlay ───────────────────────────
    draw_hud(display, len(faces), saved_count, blur_bg)

    # ── Show Frame ────────────────────────────
    cv2.imshow("🎯 Real-Time Face Detection System", display)

    # ── Key Controls ──────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        filename = f"captured_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, display)
        saved_count += 1
        print(f"✅ Image saved as: {filename}")

    elif key == ord('b') or key == ord('B'):
        blur_bg = not blur_bg
        print(f"🔵 Blur background: {'ON' if blur_bg else 'OFF'}")

    elif key == ord('q') or key == ord('Q'):
        print("👋 Quitting...")
        break

# ─────────────────────────────────────────────
# Release Resources
# ─────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print(f"✅ Session ended. Total frames: {frame_count} | Images saved: {saved_count}")
