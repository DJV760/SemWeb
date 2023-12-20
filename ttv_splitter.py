import os
import shutil
import random

# # Save the trained model
# model.save('unet_model.h5')

image_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_images'
masks_folder = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_masks'

image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.startswith('Image_')]
mask_paths = [os.path.join(masks_folder, filename) for filename in os.listdir(masks_folder) if filename.startswith('Mask_')]

image_paths.sort()
mask_paths.sort()

data_pairs = list(zip(image_paths, mask_paths))

random.shuffle(data_pairs)

num_data = len(data_pairs)
num_train = int(0.8*num_data)
num_test = int(0.1*num_data)

train_data = data_pairs[:num_train]
test_data = data_pairs[num_train:num_train+num_test]
val_data = data_pairs[num_train+num_test:]

train_dir = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\training_data'
test_dir = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\test_data'
val_dir = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\validation_data'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def copy_files(data, destination):
    for image_path, mask_path in data:
        shutil.copy(image_path, destination)
        shutil.copy(mask_path, destination)

copy_files(train_data, train_dir)
copy_files(test_data, test_dir)
copy_files(val_data, val_dir)
