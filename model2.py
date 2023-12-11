import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
import shutil
import cv2


# Load your dataset (Assuming you have X_train and y_train as your satellite images and masks)
# Make sure your images are normalized to values between 0 and 1
# You can use tools like scikit-image or OpenCV for image preprocessing

image_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_images'
mask_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_masks'

# Function to load and preprocess a single image and mask
def load_and_preprocess_image(image_path, mask_path):
    # Load images
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize images if needed (assuming they are already split into 512x512 patches)
    # If not resized during preprocessing, adjust this step accordingly
    # image = cv2.resize(image, (512, 512))
    # mask = cv2.resize(mask, (512, 512))

    # Normalize images
    image_normalized = image.astype('float32') / 255.0

    # You might need additional preprocessing steps here (e.g., data augmentation)

    return image_normalized, mask

# Lists to store images and masks
images = []
masks = []

# Iterate over the images in the folders
for filename in os.listdir(image_folder):
    if filename.startswith('Image_'):  # Assuming your images end with "_image.png"
        image_path = os.path.join(image_folder, filename)
        mask_filename = filename.replace("Image_", "Mask_")
        mask_path = os.path.join(mask_folder, mask_filename)

        # Load and preprocess images
        image, mask = load_and_preprocess_image(image_path, mask_path)

        # Append to lists
        images.append(image)
        masks.append(mask)

# Convert lists to NumPy arrays
images = np.array(images)
masks = np.array(masks)

# Split the data into training and validation sets
# You might want to adjust the test_size and random_state parameters
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42)

# Define your U-Net model
def unet_model(input_size=(512, 512, 3)):
    inputs = keras.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = layers.Concatenate(axis=3)([conv3, up5])
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = layers.Concatenate(axis=3)([conv2, up6])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = layers.Concatenate(axis=3)([conv1, up7])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    output = layers.Conv2D(1, 1, activation='sigmoid')(conv7)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


# Initialize your model
model = unet_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

train_generator = datagen.flow(train_images, train_masks, batch_size=16, seed=42)

val_generator = datagen.flow(val_images, val_masks, batch_size=16, seed=42)

model.fit(train_generator, validation_data=val_generator, epochs=10)