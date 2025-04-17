# plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_filtered_signal(filtered_signal, fs, output_path):
    """
    Plot filtered accelerometer Z-axis signal over time.
    """
    duration = len(filtered_signal) / fs
    time = np.linspace(0, duration, len(filtered_signal))

    plt.figure(figsize=(10, 4))
    plt.plot(time, filtered_signal, color='blue')
    plt.title("Filtered Accelerometer Z-Axis Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
