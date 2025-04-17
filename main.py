# # main.py

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from preprocessing import preprocess_vibro_signal
# from bp_analysis import extract_envelope, estimate_bp_points
# from plot_utils import plot_filtered_signal

# # === CONFIGURATION ===
# fs = 100  # Sampling rate (Hz)
# accel_file_path = 'test_data/acc_data_20250416.csv'  # Change filename as needed
# output_dir = 'output/'
# plot_path_filtered = os.path.join(output_dir, 'filtered_plot.png')
# plot_path_envelope = os.path.join(output_dir, 'envelope_estimation_plot.png')

# os.makedirs(output_dir, exist_ok=True)

# # === STEP 1: Load accelerometer data ===
# acc_df = pd.read_csv(accel_file_path)

# if 'timestamp' not in acc_df.columns or 'accel_z' not in acc_df.columns:
#     raise ValueError("CSV must contain 'timestamp' and 'accel_z' columns")

# timestamps = acc_df['timestamp'].values
# z_signal = acc_df['accel_z'].astype(float).values

# # === STEP 2: Filter the Z-axis signal ===
# filtered_z = preprocess_vibro_signal(z_signal, fs)

# # === STEP 3: Plot filtered signal ===
# plot_filtered_signal(filtered_z, fs, plot_path_filtered)
# print(f"[✔] Filtered Z-axis signal plotted → {plot_path_filtered}")

# # === STEP 4: Extract Envelope ===
# envelope = extract_envelope(filtered_z)

# # === STEP 5: Simulate Pressure Range (180 to 40 mmHg) ===
# pressure_range = np.linspace(180, 40, len(envelope))

# # === STEP 6: Estimate BP Points (SBP, MAP, DBP) ===
# bp_values = estimate_bp_points(envelope, pressure_range)

# # === STEP 7: Plot Envelope and BP Points ===
# plt.figure(figsize=(10, 5))
# plt.plot(pressure_range, envelope, label="Oscillometric Envelope", color='orange')
# plt.axvline(bp_values['SBP'], color='green', linestyle='--', label=f"SBP: {bp_values['SBP']} mmHg")
# plt.axvline(bp_values['MAP'], color='red', linestyle='--', label=f"MAP: {bp_values['MAP']} mmHg")
# plt.axvline(bp_values['DBP'], color='blue', linestyle='--', label=f"DBP: {bp_values['DBP']} mmHg")
# plt.gca().invert_xaxis()
# plt.xlabel("Cuff Pressure (Simulated mmHg)")
# plt.ylabel("Oscillation Amplitude")
# plt.title("Oscillometric Envelope and BP Estimation")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(plot_path_envelope)
# plt.show()

# print("\n=== Estimated Blood Pressure ===")
# print(f"🩺 Systolic (SBP):  {bp_values['SBP']} mmHg")
# print(f"🫀 Mean (MAP):      {bp_values['MAP']} mmHg")
# print(f"💤 Diastolic (DBP): {bp_values['DBP']} mmHg")


# main.py

# main.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess_vibro_signal, preprocess_ppg
from bp_analysis import extract_envelope, estimate_bp_points
from plot_utils import plot_filtered_signal

# === CONFIGURATION ===
fs_vibro = 100   # Sampling rate for accelerometer
fs_ppg = 30      # Sampling rate for PPG
accel_file_path = 'bp_smartphone/test_data/acc_data_20250416.csv'
ppg_file_path = 'bp_smartphone/test_data/ppg_data_20250417_121441.csv'

output_dir = 'output/'
os.makedirs(output_dir, exist_ok=True)
plot_path_fusion = os.path.join(output_dir, 'fused_envelope_plot.png')

# === STEP 1: Load & preprocess signals ===
acc_df = pd.read_csv(accel_file_path)
if 'timestamp' not in acc_df.columns or 'accel_z' not in acc_df.columns:
    raise ValueError("Accel CSV must contain 'timestamp' and 'accel_z' columns")

timestamps_acc = acc_df['timestamp'].values
z_signal = acc_df['accel_z'].astype(float).values
filtered_accel = preprocess_vibro_signal(z_signal, fs_vibro)
accel_envelope = extract_envelope(filtered_accel)

_, filtered_ppg = preprocess_ppg(ppg_file_path, os.path.join(output_dir, 'ppg_filtered.png'))
ppg_envelope = extract_envelope(filtered_ppg)

# === STEP 2: Align lengths ===
min_len = min(len(accel_envelope), len(ppg_envelope))
accel_env = accel_envelope[:min_len]
ppg_env = ppg_envelope[:min_len]

# === STEP 3: Normalize envelopes ===
accel_env_norm = (accel_env - np.min(accel_env)) / (np.max(accel_env) - np.min(accel_env))
ppg_env_norm = (ppg_env - np.min(ppg_env)) / (np.max(ppg_env) - np.min(ppg_env))

# === STEP 4: Weighted fusion ===
fused_envelope = 0.7 * accel_env_norm + 0.3 * ppg_env_norm

# === STEP 5: Simulate Pressure Range (180 to 40 mmHg) ===
pressure_range = np.linspace(180, 40, len(fused_envelope))

# === STEP 6: Estimate BP ===
bp_values = estimate_bp_points(fused_envelope, pressure_range)

# === STEP 7: Plot fused envelope and BP markers ===
plt.figure(figsize=(10, 5))
plt.plot(pressure_range, fused_envelope, label="Fused Envelope", color='purple')
plt.axvline(bp_values['SBP'], color='green', linestyle='--', label=f"SBP: {bp_values['SBP']} mmHg")
plt.axvline(bp_values['MAP'], color='red', linestyle='--', label=f"MAP: {bp_values['MAP']} mmHg")
plt.axvline(bp_values['DBP'], color='blue', linestyle='--', label=f"DBP: {bp_values['DBP']} mmHg")
plt.gca().invert_xaxis()
plt.xlabel("Cuff Pressure (Simulated mmHg)")
plt.ylabel("Amplitude")
plt.title("Fused Envelope (Accelerometer + PPG) - BP Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path_fusion)
plt.show()

# === Print results ===
print("\n=== Fused Signal BP Estimation ===")
print(f"🩺 Systolic (SBP):  {bp_values['SBP']} mmHg")
print(f"🫀 Mean (MAP):      {bp_values['MAP']} mmHg")
print(f"💤 Diastolic (DBP): {bp_values['DBP']} mmHg")
