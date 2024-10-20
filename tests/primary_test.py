# EEG-VR Data taken from Mohit Agarwal https://gnan.ece.gatech.edu/eeg-eyeblinks/
"""
Testing and evaluating accuracy for preprocessing and initial identification of troughs and stable points.
Model is expected to have a decently high accuracy and f1-score at this point, but a few false positive / negatives may still occur.
Hopefully, this will be fixed with the template matching and correlation steps.

* Code is largely based on read_data.py, taken from the EEG-VR dataset from Mohit Agarwal.
""" 

import sys
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.metrics import confusion_matrix

# Append the 'src' directory to the system path
src_path = os.path.join(os.path.dirname(__file__), '../src')
sys.path.append(src_path)
from preprocessing import apply_lowpass_filter
from identify_stable_points import detect_troughs, find_stable_points

# Parameters
data_folder = 'EEG-VR'

# Parameters and bandpass filtering
fs = 250.0 
def lowpass(sig, fc, fs, butter_filt_order):
    B, A = butter(butter_filt_order, np.array(fc) / (fs / 2), btype='low')
    return lfilter(B, A, sig, axis=0)

# Function to read stimulations
def decode_stim(data_path, file_stim, data_sig):
    interval_corrupt = []
    blinks = []
    n_corrupt = 0
    with open(os.path.join(data_path, file_stim)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == "corrupt":
                n_corrupt = int(row[1])
            elif n_corrupt > 0:
                if float(row[1]) == -1:
                    t_end = data_sig[-1, 0]
                else:
                    t_end = float(row[1])
                interval_corrupt.append([float(row[0]), t_end])
                n_corrupt -= 1
            elif row[0] == "blinks":
                if n_corrupt != 0:
                    print("!Error in parsing")
            else:
                blinks.append([float(row[0]), int(row[1])])
    blinks = np.array(blinks)
    return interval_corrupt, blinks

# Helper function to match detected and true blinks
def match_blinks(detected_blinks, true_blinks, tolerance=0.1):
    matched = []
    detected_flags = [0] * len(detected_blinks)
    true_flags = [0] * len(true_blinks)
    
    for i, true_blink in enumerate(true_blinks):
        for j, detected_blink in enumerate(detected_blinks):
            if abs(true_blink - detected_blink) <= tolerance:
                matched.append((true_blink, detected_blink))
                true_flags[i] = 1
                detected_flags[j] = 1 
                break
    return true_flags, detected_flags

# Load EEG Data and Ground Truth
list_of_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and '_data' in f]
file_sig = list_of_files[0]
file_stim = list_of_files[0].replace('_data', '_labels')
print("Reading:", file_sig, file_stim)

# Load data
data_sig = np.loadtxt(open(os.path.join(data_folder, file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0, 1, 2))
data_sig = data_sig[0:(int(200 * fs) + 1), :]
data_sig[:, 0] = np.array(range(0, len(data_sig))) / fs

# Load stimulations and ground truth blinks
interval_corrupt, groundtruth_blinks = decode_stim(data_folder, file_stim, data_sig)

# Apply low-pass filtering
filtered_data = np.zeros_like(data_sig)
filtered_data[:, 0] = data_sig[:, 0]
filtered_data[:, 1] = lowpass(data_sig[:, 1], 10, fs, 4)
filtered_data[:, 2] = lowpass(data_sig[:, 2], 10, fs, 4)

# Initialize lists for actual and predicted blinks (for confusion matrix)
actual_blinks_channel1 = []
predicted_blinks_channel1 = []

actual_blinks_channel2 = []
predicted_blinks_channel2 = []

# Detect troughs for both channels and validate them using stable points
for channel_index in range(1, 3):  # Loop through both EEG channels (Fp1 and Fp2)
    channel_data = filtered_data[:, channel_index]
    
    # Detect Troughs (Potential Blink Events)
    trough_indices, _ = detect_troughs(channel_data)
    print(f"Channel {channel_index}: Detected {len(trough_indices)} troughs at indices {trough_indices}")

    # Find Valid Troughs Using Stable Points
    valid_troughs = find_stable_points(channel_data, trough_indices)
    print(f"Channel {channel_index}: Detected {len(valid_troughs)} valid troughs at indices {valid_troughs}")

    # Add detected troughs to the respective predicted_blinks list for each channel
    if channel_index == 1:
        predicted_blinks_channel1.extend(filtered_data[valid_troughs, 0])
    else:
        predicted_blinks_channel2.extend(filtered_data[valid_troughs, 0])

    # Add ground truth blinks to the respective actual_blinks list for each channel
    if channel_index == 1:
        actual_blinks_channel1.extend(groundtruth_blinks[:, 0])
    else:
        actual_blinks_channel2.extend(groundtruth_blinks[:, 0])

    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data[:, 0], channel_data, label=f'Filtered EEG Channel {channel_index}')
    
    # Plot the detected valid troughs (validated blink events)
    plt.plot(filtered_data[valid_troughs, 0], channel_data[valid_troughs], 'x', label='Validated Troughs', color='red', markersize=6, alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()

    # Mark the corrupt intervals (unreliable data segments)
    for corrupt in interval_corrupt:
        plt.axvspan(corrupt[0], corrupt[1], alpha=0.5, color='red')
        if 'Corrupt Interval' not in labels:
            handles.append(plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.5))
            labels.append('Corrupt Interval')

    # Mark the ground truth blink events
    for blink in groundtruth_blinks:
        if blink[1] < 2:
            plt.axvline(x=blink[0], color='green', linestyle='--')
            if 'Ground Truth Blink' not in labels:
                handles.append(plt.Line2D([0], [0], color="green", linestyle="--"))
                labels.append('Ground Truth Blink')
        elif blink[1] == 2:
            plt.axvline(x=blink[0], color='black', linestyle='--')
            if 'Soft Blink' not in labels:
                handles.append(plt.Line2D([0], [0], color="black", linestyle="--"))
                labels.append('Soft Blink')

    plt.legend(handles, labels, loc='upper right')

    plt.title(f'EEG Channel {channel_index} - Validated Troughs vs Ground Truth Blinks')
    plt.show()

# Generate confusion matrix for Channel 1
true_flags_channel1, pred_flags_channel1 = match_blinks(predicted_blinks_channel1, actual_blinks_channel1)

conf_matrix_channel1 = confusion_matrix(true_flags_channel1, pred_flags_channel1)
print("Confusion Matrix (Channel 1):\n", conf_matrix_channel1)

# Generate confusion matrix for Channel 2
true_flags_channel2, pred_flags_channel2 = match_blinks(predicted_blinks_channel2, actual_blinks_channel2)

conf_matrix_channel2 = confusion_matrix(true_flags_channel2, pred_flags_channel2)
print("Confusion Matrix (Channel 2):\n", conf_matrix_channel2)