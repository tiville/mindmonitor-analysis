# Elizabeth on 02/04/2025 for TI research.
# Analyze 60 Hz Strength Across Electrodes from MUSE

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

# Original column names expected
channels = ["RAW_AF7", "RAW_AF8", "RAW_TP9", "RAW_TP10"]
fs = 256  # Muse 2 sampling rate

# Compute 60 Hz power safely
power_60hz = {}
for ch in channels:
    if ch in df.columns:
        signal = df[ch].dropna().values
        if np.issubdtype(signal.dtype, np.number) and signal.sum() != 0:
            try:
                power_60hz[ch] = extract_60hz_power(signal, fs)
            except Exception as e:
                print(f"Failed to process {ch}: {e}")
        else:
            print(f"{ch} has non-numeric or zero-only data — skipped.")
    else:
        print(f"{ch} not found in CSV.")

# Plot the bar chart if we have valid results
if power_60hz:
    plt.figure(figsize=(6, 4))
    plt.bar(power_60hz.keys(), power_60hz.values(), color='b')
    plt.xlabel("EEG Electrode")
    plt.ylabel("60 Hz Power (µV²)")
    plt.title("60 Hz Signal Strength at Different EEG Sites")
    plt.tight_layout()
    plt.show()

    print("60 Hz Power (µV²) by Electrode:")
    for channel, power in power_60hz.items():
        print(f"{channel}: {power:.2f}")
else:
    print("No valid EEG data found to plot.")
