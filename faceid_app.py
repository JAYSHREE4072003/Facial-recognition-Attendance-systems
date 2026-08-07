"""
Real-time face verification app.

Loads the trained Siamese model (siamesemodel.h5) and continuously reads
frames from your webcam. Press 'v' to capture the current frame and verify
it against the reference photos in application_data/verification_images/.
Press 'q' to quit.

Usage:
    python faceid_app.py
"""

import os

import cv2
import numpy as np
import tensorflow as tf

from layers import L1Dist

MODEL_PATH = "siamesemodel.h5"
INPUT_IMAGE_PATH = os.path.join("application_data", "input_image", "input_image.jpg")
VERIFICATION_DIR = os.path.join("application_data", "verification_images")

DETECTION_THRESHOLD = 0.9   # per-image similarity cutoff
VERIFICATION_THRESHOLD = 0.7  # proportion of verification images that must pass


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(VERIFICATION_DIR):
        input_img = preprocess(INPUT_IMAGE_PATH)
        validation_img = preprocess(os.path.join(VERIFICATION_DIR, image))

        result = model.predict(
            list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0
        )
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(VERIFICATION_DIR))
    verified = verification > verification_threshold

    return results, verified


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Couldn't find '{MODEL_PATH}'. Train the model first (see Faceid.ipynb) "
            "and make sure the saved .h5 file is in this directory."
        )
    if not os.path.isdir(VERIFICATION_DIR) or not os.listdir(VERIFICATION_DIR):
        raise FileNotFoundError(
            f"'{VERIFICATION_DIR}' is empty. Add a handful of reference photos of "
            "the person you want to verify before running this app."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"L1Dist": L1Dist, "BinaryCrossentropy": tf.losses.BinaryCrossentropy},
    )

    cap = cv2.VideoCapture(0)
    print("Press 'v' to verify, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imshow("Verification", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("v"):
            os.makedirs(os.path.dirname(INPUT_IMAGE_PATH), exist_ok=True)
            cv2.imwrite(INPUT_IMAGE_PATH, frame)

            _, verified = verify(model, DETECTION_THRESHOLD, VERIFICATION_THRESHOLD)
            print("VERIFIED" if verified else "NOT VERIFIED")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
