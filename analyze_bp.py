import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def bandpass_filter(data, fs=100, lowcut=0.5, highcut=10.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)

def extract_envelope(filtered_signal):
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    return envelope

def estimate_bp(envelope, pressures):
    max_amp = np.max(envelope)
    map_idx = np.argmax(envelope)
    map_pressure = pressures[map_idx]

    sbp_threshold = 0.55 * max_amp
    dbp_threshold = 0.85 * max_amp

    sbp_pressure = next((p for a, p in zip(envelope[:map_idx][::-1], pressures[:map_idx][::-1]) if a <= sbp_threshold), pressures[0])
    dbp_pressure = next((p for a, p in zip(envelope[map_idx:], pressures[map_idx:]) if a <= dbp_threshold), pressures[-1])

    return sbp_pressure, map_pressure, dbp_pressure

def analyze_vibro_file(csv_path, fs=100):
    df = pd.read_csv(csv_path)
    timestamps = df['timestamp'].values
    z_data = df['z'].values

    filtered = bandpass_filter(z_data, fs=fs)
    envelope = extract_envelope(filtered)

    pressures = np.linspace(180, 40, len(envelope))  # Simulated cuff deflation from 180 to 40 mmHg
    sbp, map_p, dbp = estimate_bp(envelope, pressures)

    print(f"Estimated SBP: {sbp:.2f} mmHg")
    print(f"Estimated MAP: {map_p:.2f} mmHg")
    print(f"Estimated DBP: {dbp:.2f} mmHg")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pressures, envelope, label='Envelope')
    plt.axvline(sbp, color='r', linestyle='--', label='SBP')
    plt.axvline(map_p, color='g', linestyle='--', label='MAP')
    plt.axvline(dbp, color='b', linestyle='--', label='DBP')
    plt.xlabel('Pressure (mmHg)')
    plt.ylabel('Oscillation Amplitude')
    plt.title('Oscillometric BP Estimation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    analyze_vibro_file("bp_smartphone/test_data/ppg_data_20250417_121441.csv")
