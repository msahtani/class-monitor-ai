import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Fearful", 2: "Neutral", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion model
json_file = open("model/emotion_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded emotion model from disk")

# Define the emotion weights
emotion_weights = {
    'Happy': 0.6,
    'Surprised': 0.5,
    'Sad': 0.3,
    'Fearful': 0.3,
    'Angry': 0.25,
    'Neutral': 0.9
}

# Start the webcam feed
# cap = cv2.VideoCapture("C:\\Users\\Honor\\Pictures\\Camera Roll\\WIN_20230723_02_48_10_Pro.jpg")
cap = cv2.VideoCapture(0)

# Create a list to store the detected emotions over time
history_emotions = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        predicted_emotion = emotion_dict[maxindex]

        # Calculate the concentration index (CI) by multiplying the emotion weights with the probability of the dominant emotion
        concentration_index = emotion_weights.get(predicted_emotion, 0) * emotion_prediction[0][maxindex]

        print(emotion_prediction[0][maxindex])
        # Determine the concentration level and text color
        if concentration_index >= 0.7:
            concentration_level = "Highly Concentrated"
            text_color = (0, 255, 0)  # Green
        elif concentration_index >= 0.4:
            concentration_level = "Nominally Concentrated"
            text_color = (0, 165, 255)  # Yellow
        else:
            concentration_level = "Not Concentrated"
            text_color = (0, 0, 255)  # Red

        # Display the percentage of concentration
        concentration_percentage = f"{int(concentration_index * 100)}%"
        cv2.putText(frame, f"Emotion: {predicted_emotion}", (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                    2, cv2.LINE_AA)
        cv2.putText(frame, f"Concentration: {concentration_level} ({concentration_percentage})", (x + 5, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        # Store the detected emotion in the history list
        history_emotions.append(predicted_emotion)

        # Limit the size of the history list to store only the last N detected emotions
        max_history_size = 30  # You can adjust this value as per your preference
        history_emotions = history_emotions[-max_history_size:]

    # while True:

    cv2.imshow('Emotion and Concentration Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
