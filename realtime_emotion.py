import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. LOAD THE MODELS AND LABELS ---

# Load the trained emotion recognition model
# The .keras format is a modern, efficient way to save models
emotion_model = load_model('emotion_model_v2.keras')

# Load the pre-trained Haar Cascade model for face detection
# This file should be in your project directory
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the emotion labels (ensure this matches the order from your training data)
# Note: If you removed a class during training, the model might not predict it.
# We'll keep all 7 for displaying, as the model's output layer expects this size.
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# --- 2. SETUP THE WEBCAM CAPTURE ---

# Start capturing video from the default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- 3. THE MAIN LOOP ---

while True:
    # Read a single frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # Convert the frame to grayscale for the Haar Cascade detector
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # detectMultiScale returns a list of rectangles for detected faces (x, y, width, height)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # --- 4. PROCESS EACH DETECTED FACE ---
    for (x, y, w, h) in faces:
        # Draw a green rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the Region of Interest (ROI) - the face - from the grayscale frame
        roi_gray = gray_frame[y:y+h, x:x+w]
        
    
        # Pre-process the ROI to match the model's input requirements
        # 1. Resize to 48x48 pixels
        roi_gray = cv2.resize(roi_gray, (48, 48))
        # 2. Convert to a NumPy array and normalize (scale pixel values to 0-1)
        roi_normalized = roi_gray.astype('float32') / 255.0
        # 3. Reshape to (1, 48, 48, 1) to create a "batch" of 1 for the model
        roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

        # --- 5. MAKE A PREDICTION ---
        # Predict the emotion using our loaded model
        prediction = emotion_model.predict(roi_reshaped)
        
        # Get the index of the highest probability (this is the predicted class)
        predicted_index = np.argmax(prediction)
        
        # Look up the corresponding emotion label from our dictionary
        predicted_label = emotion_labels[predicted_index]

        # --- 6. DISPLAY THE RESULT ---
        # Put the predicted emotion label text above the rectangle
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the final frame with detections and labels in a window
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # --- 7. EXIT CONDITION ---
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 8. CLEANUP ---
# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Application closed.")