from pipeline import load_real_time_data, preprocess_data, detect_peaks
from feature_extraction import extract_features, correlate_features
from prediction_algorithm import predict_eye_blink, calculate_power

def run_blink_detection_pipeline():
    # Step 1: Load real-time EG data
    raw_data = load_real_time_data()

    # Step 2: Preprocess data
    preprocessed_data = preprocess_data(raw_data)

    # Step 3: Detect peaks in data
    peaks = detect_peaks(preprocessed_data)

    # Step 4: Extract features
    features = extract_features(preprocessed_data, peaks)

    # Step 5: Calculate correlation and power matrices
    correlation_matrix = correlate_features(features)
    power_matrix = calculate_power(preprocessed_data)

    # Step 6: Predict eye-blink occurrence
    prediction = predict_eye_blink(correlation_matrix, power_matrix)
    return prediction

if __name__ == "__main__":
    result = run_blink_detection_pipeline()
    print(result)