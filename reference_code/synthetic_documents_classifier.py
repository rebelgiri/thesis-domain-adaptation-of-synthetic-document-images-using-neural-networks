# Reference : https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, Dropout 
from keras.models import Model, Sequential
from keras.initializers import glorot_uniform
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import cv2
import glob
import numpy as np
import PIL
from PIL import Image, ImageOps
from keras.utils import np_utils
from sklearn.utils import shuffle
import sys
import datetime
import tensorflow as tf
from tensorflow.python.keras.activations import relu, softmax

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape=(256, 256, 1), classes=10):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def preprocess_data_set(image):
    print('Start preprocess data set function')
    print(image)
    with Image.open(image) as image:
        image_grayscale = np.asarray(ImageOps.grayscale(image))
        # resizing to 256 x 256
        image_resized = cv2.resize(image_grayscale, dsize=(256 , 256), interpolation=cv2.INTER_NEAREST)
        # normalizing the images to [-1, 1]
        image_float32 = image_resized.astype('float32')
        image_normalized = (image_float32 / 127.5) - 1

    return image_normalized.reshape(256, 256, 1)

def classifier_load_data_set(data_set_path):
    print('Start load data set function')

    data_set = []
    data_set_labels = list()
    list_of_name_of_template = list()

    # Open folder, read, preprocess the images and store.
    for template_class_folder_name in os.listdir(data_set_path):
        list_of_name_of_template.append(template_class_folder_name)
        label = 0
        print(template_class_folder_name + ' is labelled as %d' %(label))
        template_class_folder_name = data_set_path +  template_class_folder_name + '/'
        image_file_names = glob.glob(template_class_folder_name + '*.png')
        
        for image in image_file_names:
            data_set.append(preprocess_data_set(image))
            data_set_labels.append(label)
        label = label + 1    

    data_set = np.asarray(data_set)
    data_set_labels = np.asarray(data_set_labels)
    # Pre-processing class labels
    data_set_labels = np_utils.to_categorical(data_set_labels, 10)
    # shuffle data set
    data_set, data_set_labels = shuffle(data_set, data_set_labels)
    print('End load data set function')

    return (data_set, data_set_labels), list_of_name_of_template

def create_model(num_classes=10):
    # model = ResNet50(input_shape=(256, 256, 1), classes=10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(256, 256, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def start_training_synthetic_documents_classifier(model, training_data_set_loader, test_data_set_loader, 
type_of_the_classifier, classes, classifier_logs, time):
     
 
    print('Start Training Classifier ' + type_of_the_classifier + '_' + time + '_model.h5', file=classifier_logs)
      
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    # test_size=0.20, random_state=42)

    log_dir = "logs/fit/" + time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = list()
    n_epochs = 3

    # Fit the model
    for _ in range(n_epochs):
        for _, data in enumerate(training_data_set_loader, 0):
            X_train_tensor, y_train_tensor = data
            X_train = X_train_tensor.numpy()
            X_train = np.einsum('ijkl->iklj', X_train)
            y_train = y_train_tensor.numpy()

            # Pre-processing class labels
            y_train = np_utils.to_categorical(y_train, 10)
            
            history.append(model.fit(X_train, y_train, epochs=1, batch_size=10, 
            validation_split=0.2, callbacks=[tensorboard_callback]))


    print('Training Finished...')
    
    # Save the results
    length = len(history)
    h = np.zeros((length, 5), dtype=np.float32)

    for i in range(length):
        h[i, 0] = i
        h[i, 1] = np.array(history[i].history['accuracy'])
        h[i, 2] = np.array(history[i].history['loss'])
        h[i, 3] = np.array(history[i].history['val_accuracy'])
        h[i, 4] = np.array(history[i].history['val_loss'])

    plt.plot(h[:, 1] * 100, '--')
    plt.plot(h[:, 2] * 100, '-.')
    plt.plot(h[:, 3] * 100, ':')
    plt.plot(h[:, 4] * 100, '-')

    plt.legend(['acc', 'loss', 'val_acc', 'val_loss'], loc='lower right')
    plt.axis([0, length, 0, 100])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Error in percentage')
    plt.grid('on')

    plt.show()
    plt.savefig(type_of_the_classifier + '_' + time + '_model.png')
    plt.close()


    '''
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('Epoch_Vs_Accuracy_Results.png')
    '''

    # Evaluate on real annoted data
    test_data_set_iter = iter(test_data_set_loader)
    X_test_tensor, y_test_tensor = test_data_set_iter.next()
    X_test = X_test_tensor.numpy()
    X_test = np.einsum('ijkl->iklj', X_test)
    y_test = y_test_tensor.numpy()
    y_test_true = y_test
    # Pre-processing class labels
    y_test = np_utils.to_categorical(y_test, 10)

    print('Evaluation Results', file=classifier_logs)
    print(X_test.shape, file=classifier_logs)
    print(y_test.shape, file=classifier_logs)

    y_test_pred = np.argmax(model.predict(X_test), axis=-1)

    results = model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_true, y_test_pred, target_names=classes, zero_division=1), file=classifier_logs)
    print("test loss, test acc:", results, file=classifier_logs)

    # serialize weights to HDF5
    model.save(type_of_the_classifier + '_' + time + '_model.h5')
    
    print('Saved model to disk ' + type_of_the_classifier + '_' + time + '_model.h5', file=classifier_logs)
    print('End Training ' + type_of_the_classifier + '_' + time + '_model.h5', file=classifier_logs)




