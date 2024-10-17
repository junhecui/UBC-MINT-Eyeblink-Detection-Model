from data_preprocessing import apply_lowpass_filter, normalize_eeg_data
from blink_detection import find_local_minima, find_stable_points
from template_matching import extract_blink_template, compute_correlation
from classification import classify_blinks

def run_blink_detection_pipeline(eeg_data):
    """
    Runs the blink detection pipeline.
    """
    # Step 1: Preprocess EEG data
    filtered_data = apply_lowpass_filter(eeg_data)
    normalized_data = normalize_eeg_data(filtered_data)

    # Step 2: Detect blink candidates
    minima = find_local_minima(normalized_data)
    stable_points = find_stable_points(normalized_data, minima)

    # Step 3: Template matching and correlation
    blink_template = extract_blink_template(normalized_data, blink_indices=[])
    correlations = compute_correlation(normalized_data, blink_template, stable_points)

    # Step 4: Classify as blink or noise
    classifications = classify_blinks(correlations)

    return classifications

if __name__ == "__main__":
    dummy_eeg_data = []
    result = run_blink_detection_pipeline(dummy_eeg_data)
    print(result)