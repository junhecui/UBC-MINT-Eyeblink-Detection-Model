# detects troughs in EEG data that could correspond to eye-blink events
# finds stable points around troughs to ensure they are valid 

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def detect_troughs(eeg_data, prominence=100, distance=10):
    """
    Detects troughs in the EEG data that could correspond to eye-blink events.
    
    Args:
        eeg_data (numpy array): Filtered EEG signal (1D array)
        prominence (float): Minimum prominence of peaks (troughs) to detect.
        distance (int): Minimum number of samples between detected troughs.
    Returns:
        trough_indices (numpy array)
    """
    # Invert to detect troughs as peaks
    inverted_signal = -eeg_data

    trough_indices, properties = signal.find_peaks(inverted_signal, prominence=prominence, distance=distance)

    return trough_indices, properties

def find_stable_points(eeg_data, trough_indices, stable_window=50, stable_threshold=50, baseline_threshold=100):
    """
    For each detected trough, check for nearby stable points where the signal recovers from the trough.
    
    Args:
        eeg_data: Filtered EEG signal (1D array)
        trough_indices (list): Indices of detected troughs.
        stable_window (int): Window size after the trough to search for stable points.
        stable_threshold (float): Threshold for determining if a point is stable.
        baseline_threshold (float): Threshold for determining of signal returns to baseline.
    Returns:
        valid_troughs (list): Troughs with nearby stable points.
    """
    valid_troughs = []

    for trough_idx in trough_indices:
        # Edge case for end of signal
        if trough_idx + stable_window >= len(eeg_data):
            continue

        """
        Creates sliding window that does not include trough in order to get baseline for recovery purposes.
        Calculates every iteration due to potential changes in baseline as signal progresses.
        """
        window_start = max(0, trough_idx - stable_window)
        window_end = trough_idx - 1
        if window_end <= window_start:
            continue
        local_baseline = np.mean(eeg_data[window_start:window_end])

        search_window = eeg_data[trough_idx:trough_idx + stable_window]

        # Calculate slope to check for stability (first derivative)
        slope = np.gradient(search_window)

        # Signal is stable if slope is near zero (under stable_threshold, adjustable) 
        stable_point_found = np.sum(np.abs(slope) < stable_threshold) > 0.9 * len(slope)

        # Check if signal recovers close to baseline (under baseline_threshold, adjustable)
        recovers_to_baseline = np.abs(search_window[-1] - local_baseline) < baseline_threshold

        # Valid trough if both conditions met
        if stable_point_found and recovers_to_baseline:
            valid_troughs.append(trough_idx)

    return valid_troughs