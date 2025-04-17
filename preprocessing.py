# preprocessing.py
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# --- Common Bandpass Filter Utility ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)

# --- For PPG File Processing ---
def load_ppg_csv(file_path):
    df = pd.read_csv(file_path)
    timestamps = df['timestamp'].values
    red_signal = df['red'].values
    return timestamps, red_signal

def preprocess_ppg(file_path, output_plot_path):
    timestamps, red_signal = load_ppg_csv(file_path)
    filtered = bandpass_filter(red_signal, lowcut=0.5, highcut=4.0, fs=30.0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, red_signal, alpha=0.4, label="Raw")
    plt.plot(timestamps, filtered, color='red', label="Filtered")
    plt.legend()
    plt.title("PPG Red Channel - Raw vs Filtered")
    plt.xlabel("Timestamp")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()

    return timestamps, filtered

# --- ✅ NEW: Accelerometer Signal Preprocessing (Missing Earlier) ---
def preprocess_vibro_signal(z_values, fs):
    """
    Bandpass filter the accelerometer Z-axis signal in the oscillometric range (0.5–5 Hz).
    """
    return bandpass_filter(z_values, lowcut=0.5, highcut=5.0, fs=fs)
