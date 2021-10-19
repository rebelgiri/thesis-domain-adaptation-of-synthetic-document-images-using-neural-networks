

import tensorflow as tf
import os

'''
class_names = ['DE_LY_Arm_2020-01', 
  'DE_LY_Bein_2018-08', 'DE_LY_Bein_2019-01',
  'DE_LY_Bein_2019-07', 'DE_LY_Bein_2020-01', 
  'DE_LY_Bein_2020-03', 'DE_LY_Hand_2020-01', 
  'DE_PH_Bein_2018-09', 'DE_PH_Bein_2019-02',
  'DE_PH_Bein_2020-01']
'''  

def normalize_img(img):
    # Map values in the range [-1, 1]
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def preprocess_cyclegan_images(image_path):
    img = tf.io.read_file(image_path)
    # Convert the image in grayscale
    img = tf.image.decode_png(img, channels=1)
    # Random flip left to right
    img = tf.image.random_flip_left_right(img)
    # Resize the image [[286, 286]]
    img = tf.image.resize(img, [286, 286])
    # Random crop image [[256, 256]]
    img = tf.image.random_crop(img, [256, 256, 1])
    img = normalize_img(img)
    return img
    
def get_label(file_path, class_names):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # Integer encode the label
  return parts[-2] == class_names

def decode_img(img):
    # Convert the image in grayscale
    img = tf.image.decode_png(img, channels=1)
    # Random flip left to right
    img = tf.image.random_flip_left_right(img)
    # Resize the image [[286, 286]]
    img = tf.image.resize(img, [286, 286])
    # Random crop image [[256, 256]]
    img = tf.image.random_crop(img, [256, 256, 1])
    img = normalize_img(img)
    return img

def preprocess_classifier_images(file_path, class_names):
  label = get_label(file_path, class_names)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


def preprocess_images_with_augmentation(file_path, class_names, 
  img_random_crop_dim, img_aug_resize_dim):

  label = get_label(file_path, class_names)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  # Convert the image in grayscale
  img = tf.image.decode_png(img, channels=1)
  # Random flip left to right
  img = tf.image.random_flip_left_right(img)
  # Resize the image Ex: [286, 286], [572, 572]
  img = tf.image.resize(img, img_aug_resize_dim)
  # Random crop image Ex: [256, 256, 1], [512, 512, 1]
  img = tf.image.random_crop(img, img_random_crop_dim)
  # Map values in the range [-1, 1]
  img = normalize_img(img)  

  return img, label    

def preprocess_images_without_augmentation(file_path, class_names, img_resize_dim):
  label = get_label(file_path, class_names)

  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  # Convert the image in grayscale
  img = tf.image.decode_png(img, channels=1)
  # Resize the image Ex: [256, 256], [512, 512]
  img = tf.image.resize(img, img_resize_dim)
  # Map values in the range [-1, 1]
  img = normalize_img(img)  
  
  return img, label  

def preprocess_numpy_array_image(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  return image, label


def configure_for_performance(ds, batch_size, buffer_size):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size, seed=10)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds    

def configure_for_performance_without_shuffle(ds, batch_size):
  ds = ds.cache()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds  