import os
import zipfile
from PIL import Image
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator

class_names = ["tree canopy", "water", "rock", "dirt", "sand", "waterlilly", "none", "swamp", "grass"]

def map_filename_to_image_and_mask(t_filename, a_filename, height=512, width=512):
    '''
    Preprocesses the dataset by:
      * resizing the input image and label maps
      * normalizing the input image pixels
      * reshaping the label maps from (height, width, 1) to (height, width, 12)

    Args:
      t_filename (string) -- path to the raw input image
      a_filename (string) -- path to the raw annotation (label map) file
      height (int) -- height in pixels to resize to
      width (int) -- width in pixels to resize to

    Returns:
      image (tensor) -- preprocessed image
      annotation (tensor) -- preprocessed annotation
    '''

    # Convert image and mask files to tensors
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_png(img_raw)
    annotation = tf.image.decode_png(anno_raw)

    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:, :, 0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))

    annotation = tf.stack(stack_list, axis=2)

    # Normalize pixels in the input image
    image = image/127.5
    image -= 1

    return image, annotation

BATCH_SIZE = 16

def get_dataset_slice_paths(image_dir, label_map_dir):
    '''
    generates the lists of image and label map paths

    Args:
      image_dir (string) -- path to the input images directory
      label_map_dir (string) -- path to the label map directory

    Returns:
      image_paths (list of strings) -- paths to each image file
      label_map_paths (list of strings) -- paths to each label map
    '''
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]

    return image_paths, label_map_paths

def get_training_dataset(image_paths, label_map_paths):
      '''
      Prepares shuffled batches of the training set.

      Args:
        image_paths (list of strings) -- paths to each image file in the train set
        label_map_paths (list of strings) -- paths to each label map in the train set

      Returns:
        tf Dataset containing the preprocessed train set
      '''
      training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
      training_dataset = training_dataset.map(map_filename_to_image_and_mask)
      training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
      training_dataset = training_dataset.batch(BATCH_SIZE)
      training_dataset = training_dataset.repeat()
      training_dataset = training_dataset.prefetch(-1)

      return training_dataset

def get_validation_dataset(image_paths, label_map_paths):
      '''
      Prepares batches of the validation set.

      Args:
        image_paths (list of strings) -- paths to each image file in the val set
        label_map_paths (list of strings) -- paths to each label map in the val set

      Returns:
        tf Dataset containing the preprocessed validation set
      '''
      validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
      validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
      validation_dataset = validation_dataset.batch(BATCH_SIZE)
      validation_dataset = validation_dataset.repeat()

      return validation_dataset

training_image_paths, training_label_map_paths = get_dataset_slice_paths('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\training_data\\train_img', 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\training_data\\train_mask')
validation_image_paths, validation_label_map_paths = get_dataset_slice_paths('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\validation_data\\val_img', 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\validation_data\\val_mask')

training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

# generate a list that contains one color for each class
colors = sns.color_palette(None, len(class_names))

# print class name - normalized RGB tuple pairs
# the tuple values will be multiplied by 255 in the helper functions later
# to convert to the (0,0,0) to (255,255,255) RGB values you might be familiar with
for class_name, color in zip(class_names, colors):
    print(f'{class_name} -- {color}')

# Visualization Utilities

def fuse_with_pil(images):
      '''
      Creates a blank image and pastes input images

      Args:
        images (list of numpy arrays) - numpy array representations of the images to paste

      Returns:
        PIL Image object containing the images
      '''

      widths = (image.shape[1] for image in images)
      heights = (image.shape[0] for image in images)
      total_width = sum(widths)
      max_height = max(heights)

      new_im = PIL.Image.new('RGB', (total_width, max_height))

      x_offset = 0
      for im in images:
        pil_image = PIL.Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset,0))
        x_offset += im.shape[1]

      return new_im


def give_color_to_annotation(annotation):
      '''
      Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
      the third axis represents the color channel. The label values are multiplied by
      255 and placed in this axis to give color to the annotation

      Args:
        annotation (numpy array) - label map array

      Returns:
        the annotation array with an additional color channel/axis
      '''
      seg_img = np.zeros((annotation.shape[0], annotation.shape[1], 3)).astype('float')

      for c in range(len(class_names)):
        segc = (annotation == c)
        seg_img[:, :, 0] += segc*(colors[c][0] * 255.0)
        seg_img[:, :, 1] += segc*(colors[c][1] * 255.0)
        seg_img[:, :, 2] += segc*(colors[c][2] * 255.0)

      return seg_img


def show_predictions(image, labelmaps, titles, iou_list, dice_score_list):
      '''
      Displays the images with the ground truth and predicted label maps

      Args:
        image (numpy array) -- the input image
        labelmaps (list of arrays) -- contains the predicted and ground truth label maps
        titles (list of strings) -- display headings for the images to be displayed
        iou_list (list of floats) -- the IOU values for each class
        dice_score_list (list of floats) -- the Dice Score for each class
      '''

      true_img = give_color_to_annotation(labelmaps[1])
      pred_img = give_color_to_annotation(labelmaps[0])

      image = image + 1
      image = image * 127.5
      images = np.uint8([image, pred_img, true_img])

      metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
      metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

      display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) for idx, iou, dice_score in metrics_by_id]
      display_string = "\n\n".join(display_string_list)

      plt.figure(figsize=(15, 4))

      for idx, im in enumerate(images):
            plt.subplot(1, 3, idx+1)
            if idx == 1:
                plt.xlabel(display_string)
            plt.xticks([])
            plt.yticks([])
            plt.title(titles[idx], fontsize=12)
            plt.show(im)


def show_annotation_and_image(image, annotation):
      '''
      Displays the image and its annotation side by side

      Args:
        image (numpy array) -- the input image
        annotation (numpy array) -- the label map
      '''
      new_ann = np.argmax(annotation, axis=2)
      seg_img = give_color_to_annotation(new_ann)

      image = image + 1
      image = image * 127.5
      image = np.uint8(image)
      # images = [image, seg_img]

      images = [image, seg_img]
      fused_img = fuse_with_pil(images)
      plt.imshow(fused_img)
      plt.show()


def list_show_annotation(dataset):
      '''
      Displays images and its annotations side by side

      Args:
        dataset (tf Dataset) - batch of images and annotations
      '''

      ds = dataset.unbatch()
      ds = ds.shuffle(buffer_size=100)

      plt.figure(figsize=(25, 15))
      plt.title("Images And Annotations")
      plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

      # we set the number of image-annotation pairs to 9
      # feel free to make this a function parameter if you want
      for idx, (image, annotation) in enumerate(ds.take(9)):
            plt.subplot(3, 3, idx + 1)
            plt.yticks([])
            plt.xticks([])
            show_annotation_and_image(image.numpy(), annotation.numpy())

list_show_annotation(training_dataset)
list_show_annotation(validation_dataset)

def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
      '''
      Defines a block in the VGG network.

      Args:
        x (tensor) -- input image
        n_convs (int) -- number of convolution layers to append
        filters (int) -- number of filters for the convolution layers
        activation (string or object) -- activation to use in the convolution
        pool_size (int) -- size of the pooling layer
        pool_stride (int) -- stride of the pooling layer
        block_name (string) -- name of the block

      Returns:
        tensor containing the max-pooled output of the convolutions
      '''

      for i in range(n_convs):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same', name="{}_conv{}".format(block_name, i + 1))(x)

      x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, name="{}_pool{}".format(block_name, i+1))(x)

      return x

vgg_weights_path = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG_16(image_input):
      '''
      This function defines the VGG encoder.

      Args:
        image_input (tensor) - batch of images

      Returns:
        tuple of tensors - output of all encoder blocks plus the final convolution layer
      '''

      # create 5 blocks with increasing filters at each stage.
      # you will save the output of each block (i.e. p1, p2, p3, p4, p5). "p" stands for the pooling layer.
      x = block(image_input,n_convs=2, filters=64, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block1')
      p1= x

      x = block(x,n_convs=2, filters=128, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block2')
      p2 = x

      x = block(x,n_convs=3, filters=256, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block3')
      p3 = x

      x = block(x,n_convs=3, filters=512, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block4')
      p4 = x

      x = block(x,n_convs=3, filters=512, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block5')
      p5 = x

      # create the vgg model
      vgg  = tf.keras.Model(image_input , p5)

      # load the pretrained weights you downloaded earlier
      vgg.load_weights(vgg_weights_path)

      # number of filters for the output convolutional layers
      n = 4096

      # our input images are 224x224 pixels so they will be downsampled to 7x7 after the pooling layers above.
      # we can extract more features by chaining two more convolution layers.
      c6 = tf.keras.layers.Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6")(p5)
      c7 = tf.keras.layers.Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7")(c6)

      # return the outputs at each stage. you will only need two of these in this particular project
      # but we included it all in case you want to experiment with other types of decoders.
      return (p1, p2, p3, p4, c7)

def fcn8_decoder(convs, n_classes):
      '''
      Defines the FCN 8 decoder.

      Args:
        convs (tuple of tensors) - output of the encoder network
        n_classes (int) - number of classes

      Returns:
        tensor with shape (height, width, n_classes) containing class probabilities
      '''

      # unpack the output of the encoder
      f1, f2, f3, f4, f5 = convs

      # upsample the output of the encoder then crop extra pixels that were introduced
      o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False )(f5)
      o = tf.keras.layers.Cropping2D(cropping=(1,1))(o)

      # load the pool 4 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
      o2 = f4
      o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)

      # add the results of the upsampling and pool 4 prediction
      o = tf.keras.layers.Add()([o, o2])

      # upsample the resulting tensor of the operation you just did
      o = (tf.keras.layers.Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False ))(o)
      o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

      # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
      o2 = f3
      o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)

      # add the results of the upsampling and pool 3 prediction
      o = tf.keras.layers.Add()([o, o2])

      # upsample up to the size of the original image
      o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False )(o)

      # append a softmax to get the class probabilities
      o = (tf.keras.layers.Activation('softmax'))(o)

      return o

def segmentation_model():
      '''
      Defines the final segmentation model by chaining together the encoder and decoder.

      Returns:
        keras Model that connects the encoder and decoder networks of the segmentation model
      '''

      inputs = tf.keras.layers.Input(shape=(512, 512, 3,))
      convs = VGG_16(image_input=inputs)
      outputs = fcn8_decoder(convs, 9)
      model = tf.keras.Model(inputs=inputs, outputs=outputs)

      return model

model = segmentation_model()
model.summary()

sgd = tf.keras.optimizers.SGD(lr=1E-2, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# number of training images
train_count = 674

# number of validation images
validation_count = 68

EPOCHS = 50

steps_per_epoch = train_count//(BATCH_SIZE*2)
validation_steps = validation_count//(BATCH_SIZE*2)

history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps, epochs=EPOCHS)

model.save_weights('model_third_parameteres.h5')
