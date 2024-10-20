import numpy as np
from preprocessing import apply_lowpass_filter
from identify_stable_points import detect_troughs, find_stable_points
from src.correlation import compute_correlation, high_corr_comp

def blink_detection_pipeline(eeg_data):
    """
    Executes the blink detection pipeline.
    """