#Elizabeth on 02/04/2025 for TI research.
#Steps to Analyze 60 Hz Strength Across Electrodes from MUSE
#Record EEG with MUSE/Mind Monitor → Export CSV file.
#Use FFT (Fast Fourier Transform) to isolate 60 Hz activity.
#Compare 60 Hz amplitude between AF7, AF8, TP9, TP10:
#If AF7 > AF8 → The signal is stronger on the left forehead.
#If AF8 > AF7 → The signal is stronger on the right forehead.
#If TP9 > TP10 → The signal is stronger on the left temporal lobe.
#If TP10 > TP9 → The signal is stronger on the right temporal lobe.
#Create a heatmap to visualize signal strength.
#Mind Monitor Data Example (CSV Format)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Function to extract 60 Hz power using Welch's method
def extract_60hz_power(signal, fs):
    freqs, psd = welch(signal, fs, nperseg=fs*2)
    return psd[np.argmin(np.abs(freqs - 60))]

# Load Mind Monitor CSV file
file_path = input("Enter the full path to your Mind Monitor CSV file: ")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("The file was not found. Please check the path and try again.")
    exit()

# Select EEG columns
channels = ["RAW_AF7", "RAW_AF8", "RAW_TP9", "RAW_TP10"]
fs = 256  # Muse 2 sampling rate

# Check if required columns are present
missing_channels = [ch for ch in channels if ch not in df.columns]
if missing_channels:
    print(f"Missing columns in the CSV file: {missing_channels}")
    exit()

# Compute 60 Hz power for each channel
power_60hz = {ch: extract_60hz_power(df[ch].dropna(), fs) for ch in channels}

# Plot a heatmap
plt.figure(figsize=(6, 4))
plt.bar(power_60hz.keys(), power_60hz.values(), color='b')
plt.xlabel("EEG Electrode")
plt.ylabel("60 Hz Power (µV²)")
plt.title("60 Hz Signal Strength at Different EEG Sites")
plt.show()

# Print results
print("60 Hz Power (µV²) by Electrode:")
for channel, power in power_60hz.items():
    print(f"{channel}: {power:.2f}")
