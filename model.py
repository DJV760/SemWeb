import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import shutil


# Load your dataset (Assuming you have X_train and y_train as your satellite images and masks)
# Make sure your images are normalized to values between 0 and 1
# You can use tools like scikit-image or OpenCV for image preprocessing

# Define your U-Net model
# def unet_model(input_size=(512, 512, 3)):
#     inputs = keras.Input(input_size)
#
#     # Encoder
#     conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
#     pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
#     conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
#     pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
#     conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
#     pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     # Bottom
#     conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
#     conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
#
#     # Decoder
#     up5 = layers.UpSampling2D(size=(2, 2))(conv4)
#     up5 = layers.Conv2D(256, 2, activation='relu', padding='same')(up5)
#     merge5 = layers.Concatenate(axis=3)([conv3, up5])
#     conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
#     conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
#
#     up6 = layers.UpSampling2D(size=(2, 2))(conv5)
#     up6 = layers.Conv2D(128, 2, activation='relu', padding='same')(up6)
#     merge6 = layers.Concatenate(axis=3)([conv2, up6])
#     conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
#     conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
#
#     up7 = layers.UpSampling2D(size=(2, 2))(conv6)
#     up7 = layers.Conv2D(64, 2, activation='relu', padding='same')(up7)
#     merge7 = layers.Concatenate(axis=3)([conv1, up7])
#     conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
#     conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
#
#     # Output layer
#     output = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
#
#     model = keras.Model(inputs=inputs, outputs=output)
#     return model
#
#
# # Initialize your model
# model = unet_model()



img_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_images'
all_images = os.listdir(img_folder)

train_images, test_val_images = train_test_split(all_images, test_size=0.2, random_state=42)
test_images, val_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

train_img_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\train_img_folder'
test_img_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\test_img_folder'
val_img_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\val_img_folder'

os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(test_img_folder, exist_ok=True)
os.makedirs(val_img_folder, exist_ok=True)

# Move images to respective folders
for img in train_images:
    shutil.move(os.path.join(img_folder, img), os.path.join(train_img_folder, img))

for img in test_images:
    shutil.move(os.path.join(img_folder, img), os.path.join(test_img_folder, img))

for img in val_images:
    shutil.move(os.path.join(img_folder, img), os.path.join(val_img_folder, img))

# mask_folder = "C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_masks"
# all_masks = os.listdir(mask_folder)
#
# train_masks, test_val_masks = train_test_split(all_masks, test_size=0.2, random_state=42)
# test_masks, val_masks = train_test_split(test_val_masks, test_size=0.5, random_state=42)
#
# train_mask_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\train_mask_folder'
# test_mask_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\test_mask_folder'
# val_mask_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\val_mask_folder'
#
# os.makedirs(train_mask_folder, exist_ok=True)
# os.makedirs(test_mask_folder, exist_ok=True)
# os.makedirs(val_mask_folder, exist_ok=True)
#
# # Move images to respective folders
# for mask in train_masks:
#     shutil.move(os.path.join(mask_folder, mask), os.path.join(train_mask_folder, mask))
#
# for mask in test_masks:
#     shutil.move(os.path.join(mask_folder, mask), os.path.join(test_mask_folder, mask))
#
# for mask in val_masks:
#     shutil.move(os.path.join(mask_folder, mask), os.path.join(val_mask_folder, mask))


# # Use ImageDataGenerator to load and preprocess images and masks
# image_datagen = ImageDataGenerator(rescale=1./255)
# mask_datagen = ImageDataGenerator(rescale=1./255)
#
# # Set up data generators
# image_generator = image_datagen.flow_from_directory(
#     img_folder,
#     class_mode=None,
#     target_size=(512, 512),
#     batch_size=16,
#     seed=1
# )
#
# mask_generator = mask_datagen.flow_from_directory(
#     mask_folder,
#     class_mode=None,
#     target_size=(512, 512),
#     batch_size=16,
#     seed=1
# )
#
# # Combine image and mask generators
# train_generator = zip(image_generator, mask_generator)
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator), validation_data=val_generator, validation_steps=len(val_generator))
#
# # Save the trained model
# model.save('unet_model.h5')
