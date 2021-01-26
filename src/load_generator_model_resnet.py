# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os
from PIL import Image
import numpy as np


def normalize(image):
  # normalizing the images to [-1, 1]
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def preprocess_test_images(image):
    print(image)
    
    
    
    
    # image = convert_image_in_rgb(image)
    # resizing to 256 x 256 x 3
    # image = tf.image.resize(image, [256, 256],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = normalize(image)
    # return image  

def convert_image_in_rgb(image):
    # Convert into RGB image(3D) only if image is a Gray scale image (2D)
    with Image.open(image) as img:
        number_of_dimetions = len(img.size)
        if(2 == number_of_dimetions):
            image = np.asarray(img.convert('RGB'))
    return image   


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
        pyplot.savefig('Results_From_ResNet_Generator.png')
    pyplot.close()

 # Load and preprocess test data set images
def load_test_data_set(path):
    print('Start: Load and preprocess test data set images')
    test_data_set = []
    for f in os.listdir(path + 'test/'):
        test_data_set.append(preprocess_test_images(path + 'test/' + f))

    test_data_set = np.asarray(test_data_set)
    print(test_data_set.shape)
    print('End: Load and preprocess test data set images')
    return test_data_set

# load model

# model = load_model('/home/giriraj/thesis-domain-adaptation-of-synthetic-generated-documents/thesis-domain-adaptation-of-synthetic-generated-documents/src/CycleGAN_RESNET_Results_26_01_21/generator_model_g_model_AtoB_050.h5', 
# custom_objects={'InstanceNormalization':InstanceNormalization})

path = '/home/giriraj/giri/data/'
test_data_set = load_test_data_set(path)

# generate images
# X = model.predict(test_data_set)
# plot the result
# save_plot(X, 5)