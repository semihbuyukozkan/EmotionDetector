import cv2
import numpy as np
import os
import pretrained_model
import tensorflow as tf
import tensorflow_hub as hub

model_path = 'C:\TensorflowModels\emotion_detection_model.h5'
if not os.path.exists(model_path):
    print("Creatıng new model...")
    pretrained_model.model_creator()
else:
    print("Model file found. The model is loading...")

model = tf.keras.models.load_model(model_path, compile=True, custom_objects={'KerasLayer': hub.KerasLayer})

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (224, 224))
        face_array = np.expand_dims(face_resized, axis=0) / 255.0

        predictions = model.predict(face_array)
        emotion = emotions[np.argmax(predictions)]

        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Press Q to exit")