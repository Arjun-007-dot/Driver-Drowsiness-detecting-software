# Import Dependencies
import json
import cv2
from scipy.spatial import distance
import dlib
from imutils import face_utils
import os
import sys

# Define the Path to the Config File
PATH_TO_CONFIG_FILE = "config.json"

def load_config():
    """
    Load configuration parameters from the config.json file.
    """
    if not os.path.exists(PATH_TO_CONFIG_FILE):
        print(f"Error: {PATH_TO_CONFIG_FILE} not found.")
        sys.exit(1)

    with open(PATH_TO_CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # Ensure all required keys are present
    required_keys = ["EAR_threshold", "MAR_threshold"]
    for key in required_keys:
        if key not in config:
            print(f"Error: {key} missing in {PATH_TO_CONFIG_FILE}.")
            sys.exit(1)

    return (
        config["EAR_threshold"],
        config["MAR_threshold"]
    )

def calculate_eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.
    """
    EAR1 = distance.euclidean(eye[1], eye[5])
    EAR2 = distance.euclidean(eye[2], eye[4])
    EAR3 = distance.euclidean(eye[0], eye[3])
    return (EAR1 + EAR2) / (2.0 * EAR3)

def calculate_mouth_aspect_ratio(mouth):
    """
    Calculate the Mouth Aspect Ratio (MAR) for a given mouth.
    """
    MAR1 = distance.euclidean(mouth[13], mouth[19])
    MAR2 = distance.euclidean(mouth[14], mouth[18])
    MAR3 = distance.euclidean(mouth[15], mouth[17])
    MAR4 = distance.euclidean(mouth[12], mouth[16])
    return (MAR1 + MAR2 + MAR3) / (3.0 * MAR4)

def driver_sleep_detector():
    """
    Main function to detect driver drowsiness using EAR and MAR.
    """
    EAR_threshold, MAR_threshold = load_config()

    # Check if the dlib model file exists
    dlib_model_path = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(dlib_model_path):
        print(f"Error: {dlib_model_path} not found.")
        sys.exit(1)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_model_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Closing the System...")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            left_EAR = calculate_eye_aspect_ratio(left_eye)
            right_EAR = calculate_eye_aspect_ratio(right_eye)
            EAR = (left_EAR + right_EAR) / 2.0
            MAR = calculate_mouth_aspect_ratio(mouth)

            # Draw contours around eyes and mouth
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            mouth_hull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

            # Display EAR and MAR on the frame
            cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Driver Sleep Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    driver_sleep_detector()