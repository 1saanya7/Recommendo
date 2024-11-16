# Import libraries
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get rid of warnings
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Activation
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import itertools

print("Libraries importing done")

# Define paths
folder_path = "C:/Users/hp/Desktop/Projects/movie_recommendation/data/archive (1)"
train_dir = "C:/Users/hp/Desktop/Projects/movie_recommendation/data/archive (1)/train"
val_dir = "C:/Users/hp/Desktop/Projects/movie_recommendation/data/archive (1)/test"


# Define image size and class names
img_size = 48
emotion_list = ['0', '1', '2', '3', '4', '5', '6']  # Expected classes
num_classes = len(emotion_list)  # Number of classes

# Display sample images from each class
plt.figure(figsize=(18, 22))
i = 1
for expression in emotion_list:
    img = load_img(os.path.join(train_dir, expression, os.listdir(os.path.join(train_dir, expression))[0]))
    plt.subplot(1, 7, i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()

# Data generators for training and validation data
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_val = ImageDataGenerator(rescale=1./255)

# Define batch size
batch_size = 64

# Training data
train_set = datagen_train.flow_from_directory(
    directory=train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode="categorical",
    shuffle=True
)

# Validation data
val_set = datagen_val.flow_from_directory(
    directory=val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode="categorical",
    shuffle=False
)

# Check if the number of classes matches the model output
print(f"Expected number of classes: {num_classes}")
print(f"Number of classes in train set: {train_set.num_classes}")
print(f"Number of classes in validation set: {val_set.num_classes}")

# Confirm classes in the dataset match the model’s output layer
if train_set.num_classes != num_classes or val_set.num_classes != num_classes:
    raise ValueError("Mismatch between model output classes and dataset classes. Check the folder structure and class labels.")

print('Train and Validation sets have been created.')

# Build the model
model = Sequential()

# First layer
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second layer
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third layer   
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fourth layer
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected layers
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.30))

# Output layer (with 7 units for 7 classes)
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print('CNN model has been created. You can proceed to train your data with this model.')
model.summary()

# Train the model
history = model.fit(
    train_set,
    steps_per_epoch=train_set.n // train_set.batch_size,
    validation_data=val_set,
    validation_steps=val_set.n // val_set.batch_size,
    epochs=50
)

model.save("mood_train_3.h5")
print('Your model has been trained and saved as mood_train_3.h5!')
# Create plots for accuracy and loss.
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

