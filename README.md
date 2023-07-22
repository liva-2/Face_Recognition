# Attendance System
We are going to build a model that recognizes faces
* ## Imports.
First, we are going to import all the libraries we need to preprocess and train the data:
```python
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
* ## Load and split the data.
The first step is always loading the data, then preprocessing the data, in this case, we are going to make all the images size the same:
```python
train_dir="Original Images/"
generator = ImageDataGenerator()
train_ds = generator.flow_from_directory(train_dir,target_size=(224, 224),batch_size=32)
classes = list(train_ds.class_indices.keys())
```
* ## Build the model.
now comes the turn to build the model, we used CNN to train our data, in our case, we build 5 Convolutional layers and 5 pooling layers, and we used the linear activation function in each convolution. Finally, in the last step we convert the images into a vector so we can deel with it using flatten and dense:
```python
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(len(classes),activation='softmax'))
```
* ## Compile and Summary.
After we built the model we have to set the loss function, the optimizer, and the metric, then we are going to print a summary of the model we are trying to fit:
```python
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ["accuracy"])
model.summary()
```
* ## Training the model.
This is the waited and importing thing which is the training step, now we are going to fit the model and sit the epochs, batch size and wait till the model finish training:
```python
history = model.fit(train_ds,epochs= 30, batch_size=32)
```
* ## Save the model.
The final step now is saving the model to use it whenever we want:
```python
model.save('attend.h5')
```
* ## Importing the libraries and loading the model.
After saving the model we are going to use it right now to build our attendance system so let's import the libraries and load the model:
```python
# Import necessary libraries
import cv2
import datetime
import numpy as np
import csv
import os
from tensorflow.keras.models import load_model
import face_recognition

# Load the face recognition model
model = load_model('attend.h5')
```
* ## Load & preprocess the data.
Get your dataset ready right now cause we are going to load it and preprocess it:
```python
# Initialize empty lists to store face encodings and corresponding names
faces = []
names = []

# Loop through each image in the 'faces' directory, resize it, and encode it using the loaded model
for i in os.listdir('faces'):
    # Check if the file is a valid image file
    split_name = os.path.splitext(i)
    if split_name[1] not in ['.jpeg', '.jpg', '.png']:
        print(f"Skipping invalid file: {i}")
        continue
    # Load the image file
    image = cv2.imread(os.path.join('faces', i))
    if image is None:
        print(f"Failed to load image: {i}")
        continue
    # Resize the image to match the input size of the loaded model
    if image.shape[:2] != (224, 224):
        image = cv2.resize(image, (224, 224))
    # Expand the dimensions of the image array and normalize the pixel values
    face_img = np.expand_dims(image, axis=0)
    face_img = face_img / 255.0
    # Encode the face using the loaded model
    face_encoding = model.predict(face_img)
    # Append the encoded face and corresponding name to the lists
    faces.append(face_encoding)
    names.append(split_name[0])
```
* ## CSV File.
Create your CSV file that will save the info of the attendees:
```python
# Create a new CSV file to record attendance
with open('attendance.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Time', 'Image File'])
    
# Initialize an empty list to keep track of recorded names
recorded_names = []
```
* ## Setting the frame.
After doing all these steps we going to set the frame and the camera ready:
```python
# Open the video capture device and set the frame size and encoding
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Check if the video capture device was successfully opened
if not cap.isOpened():
    print("Failed to open video capture")
    exit(1)
```
* ## Video frame loop.
Now we are going to read the frame, preprocess the data, detect the face, recognize the face, attend and capture the recognized face, and all these steps simply clarify in comments:
```python
# Start the main loop to capture video frames and recognize faces
while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from video capture")
        break

    # Convert the BGR color format to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings in the RGB frame using the face_recognition library
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face and recognize it based on the stored encodings
    for face_location in face_locations:
        # Extract the coordinates of the face bounding box
        top, right, bottom, left = face_location
        # Extract the face image from the frame and resize it to match the input size of the loaded model
        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, (224, 224))
        # Expand the dimensions of the face image array and normalize the pixel values
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0
        # Encode the face using the loaded model
        face_encoding = model.predict(face_img)
        # Compute the Euclidean distance between the encoded face and each stored encoding
        matches = []
        for known_face_encoding in faces:
            distance = np.linalg.norm(face_encoding - known_face_encoding)
            # If the distance is below a certain threshold, the face is considered a match
            if distance < 0.6:
                matches.append(True)
            else:
                matches.append(False)
        # Set the name of the recognized face to "Unknown" by default
        name = "Unknown"
        # If there is a match, record the attendance and save a capture of the face image
        if True in matches:
            index = matches.index(True)
            name = names[index]
            if name not in recorded_names:
                # Generate a unique filename based on the current date and time
                filename = f"{name}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                # Save the capture of the face image
                cv2.imwrite(filename, frame)
                # Record the attendance in the CSV file
                with open('attendance.csv', 'a', newline='') as file:
                    writer= csv.writer(file)
                    writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename])
                # Add the name to the list of recorded names to avoid duplicates
                recorded_names.append(name)

        # Draw a rectangle around the face bounding box and display the recognized name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the video frame with recognized faces
    cv2.imshow('Attendance System', frame)

    # Wait for a key press and check if the 'q' or 's' key was pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save a capture of the current frame with a unique filename based on the current date and time
        filename = f"capture{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved capture to {filename}")
```
* ## Release and close windows.
This the last two lines before we are going run the code and the purpose of these two lines are to close the windows when we are done form the program:
```python
# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
```
