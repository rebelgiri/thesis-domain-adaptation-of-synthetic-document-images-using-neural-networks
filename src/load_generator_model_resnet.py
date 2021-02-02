# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os
from os import path
from PIL import Image, ImageOps
import numpy as np
import cv2

# normalizing the images to [-1, 1]
def normalize(image):
  image = image.astype('float32')
  image = (image / 127.5) - 1
  return image

def preprocess_test_images(image_path):
    print(image_path)
    image = Image.open(image_path)
    image_grayscale = np.asarray(ImageOps.grayscale(image))
    image_resized = cv2.resize(image_grayscale, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    image_normalized = normalize(image_resized)
    return image_normalized.reshape(256, 256, 1)

def save_images_in_larger_dimension(generated_images):
    for i in range(generated_images.shape[0]):
        image_resized = cv2.resize(generated_images[i], dsize=(2700, 3700), 
        interpolation=cv2.INTER_NEAREST)
        pyplot.axis('off')
        pyplot.imshow(generated_images[i], cmap='gray')
        pyplot.savefig('Generated_Image_%d.png' % (i))




# create and save a plot of generated images (reversed grayscale)
def save_plot(test_data_set, examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(1, 2, 1)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(test_data_set[i], cmap='gray')

        pyplot.subplot(1, 2, 2)
         # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i], cmap='gray')

        pyplot.savefig('Results_From_ResNet_Generator_%d.png' % (i))
        pyplot.close()


def load_test_data_set(path):

    print('Start function test load dataset')
    # preprocess images
    test_data_set = []
   
    for f in os.listdir(path):
        test_data_set.append(preprocess_test_images(path + f))

    test_data_set = np.asarray(test_data_set)
    print(test_data_set.shape)
    print('End function load test data set')
    return test_data_set

# load model

model = load_model('generator_model_g_model_AtoB_045.h5',
custom_objects={'InstanceNormalization': InstanceNormalization})

path = '/home/giriraj/giri/data/synthetic_document_images_test/documents/'
test_data_set = load_test_data_set(path)

# generate images
X = model.predict(test_data_set)

save_images_in_larger_dimension(X)

# plot the result
save_plot(test_data_set, X, 5)