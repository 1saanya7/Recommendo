import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('mood_train_3.h5')

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the list of emotion labels in the same order as the model's output
emotion_list = ['0', '1', '2', '3', '4', '5', '6']  # Replace with actual emotion names if applicable

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangles around detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))  # Resize to match model's input size
        face = face.astype('float32') / 255  # Normalize pixel values
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)  # Add batch dimension (1, 48, 48, 1)

        # Predict the mood
        prediction = model.predict(face)
        mood_index = np.argmax(prediction)  # Get index of the highest confidence value
        mood_label = emotion_list[mood_index]
        print(mood_label)
        cv2.putText(frame, mood_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print("Please Choose between Movie, Song, Books")
        choice = input("Enter your choice:")
        if(choice.lower()=='movie'):
            import movie_recommend
            predictions = movie_recommend.recommend_movies(mood_label, movie_recommend.mood_genre_mapping,movie_recommend.genre_to_dataframe )
            print(predictions)
        elif(choice.lower()=='song'):
            import song_recommend
            predictions = song_recommend.recommend_songs_based_on_mood(mood_label)
            print(predictions)
        elif(choice.lower()=='book'):
            import book_recommend
            predictions = book_recommend.recommend_books_based_on_mood(mood_label)
            print(predictions)
        else:
            print("Make a valid choice")

    

    cv2.imshow('Mood Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break
cap.release()
cv2.destroyAllWindows()
