import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.signal import find_peaks, detrend  # <- Added detrend
import matplotlib.pyplot as plt

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print("FPS:", fps)

# Data storage
brightness_values = []
timestamps = []

# Monitoring time
duration = 40  # seconds
start_time = time.time()

# Moving average window size
window_size = 3

def estimate_heart_rate(signal, fps):
    # Apply moving average
    smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    smoothed -= np.mean(smoothed)

    # Find peaks
    peaks, _ = find_peaks(smoothed, distance=fps * 0.45)
    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fps
    avg_interval = np.mean(intervals)
    bpm = 60 / avg_interval
    return round(bpm, 2)

while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        detection = results.detections[0]
        ih, iw, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

        # Forehead region
        forehead = frame[y:y + h//5, x + w//4:x + 3*w//4]
        if forehead.size != 0:
            brightness = np.mean(cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY))
            brightness_values.append(brightness)
            timestamps.append(time.time() - start_time)

        # Draw box & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Live HR Estimation
        if len(brightness_values) > fps * 5:
            # Apply detrending for HR calculation
            corrected_signal = detrend(brightness_values[-int(fps*10):])
            hr = estimate_heart_rate(corrected_signal, fps)
            if hr:
                cv2.putText(frame, f"HR: {hr} BPM", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Calculating HR...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Heart Rate Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Final HR Output
print("\n⏱ 40 seconds done.")
if len(brightness_values) > fps * 10:
    corrected_signal = detrend(brightness_values)
    final_hr = estimate_heart_rate(corrected_signal, fps)
    if final_hr:
        print(f"✅ Final Estimated Heart Rate: {final_hr} BPM")
    else:
        print("❌ Could not determine a valid heart rate. Try again.")
else:
    print("❌ Not enough data collected.")

# Plot PPG signal (with baseline corrected)
if brightness_values:
    corrected_signal = detrend(brightness_values)

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps[:len(corrected_signal)], corrected_signal, label="Corrected PPG Signal", color='purple')
    plt.xlabel("Time (s)")
    plt.ylabel("Brightness (corrected)")
    plt.title("PPG Signal After Baseline Drift Correction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
