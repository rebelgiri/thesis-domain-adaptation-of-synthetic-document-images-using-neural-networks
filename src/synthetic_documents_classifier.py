import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import load_img
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pickle
import joblib
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_dataset():
    print('Start load data set function')


    print('End load data set function')


def main():
    
    print('Inside main function')
    # Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = load_dataset()

    # Pre-processing input data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test  = X_test.astype('float32')
    X_test  /= 255

    # Pre-processing class labels
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Construct convolution neural network
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', use_bias=True,
                            input_shape=(28, 28, 1)))
    model.add(Convolution2D(32, (3, 3), activation='relu', use_bias=True,
                            input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    print(model.output_shape)
    model.summary()

    # Compile model
    print(model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']))

    # Fit the model
    history = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))

    # Save the results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('Epoch_Vs_Accuracy_Results.png')

    # Evaluate on real annoted data
    results = model.evaluate(X_test, y_test, verbose=2)
    print("test loss, test acc:", results)

    # serialize model to JSON
    # model_json = model.to_json()
    # with open("synthetic_documents_classifier.json", "w") as json_file:
    #    json_file.write(model_json)

    # serialize weights to HDF5
    model.save("synthetic_documents_classifier_model.h5")
    # print("Saved model to disk")

    # ynew = np.argmax(model.predict_classes(X_test), axis=-1)
    # print('End')


if __name__ == "__main__":
    path =''
    data_set =  load_data_set(path)
    start_training(data_set)
