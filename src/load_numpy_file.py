

from keras.models import load_model
from classifier_model import *
from train_synthetic_documents_classifier import *
import sys
from datetime import datetime
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE
from preprocessing_images import *
from numpy import load, save



if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    cyclegan_predicted_images_path = sys.argv[1]
    cyclegan_predicted_labels_path = sys.argv[2]
    classifier_test_data_set_path = sys.argv[3]
    classifier_name = sys.argv[4]

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    path = classifier_name + '_' + time
    os.mkdir(path)
    print("Directory '% s' created" % path)
    path = path + '/'

    test_data_dir = pathlib.Path(classifier_test_data_set_path)
    test_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*'), shuffle=False)

    class_names = np.array(sorted([item.name for item in test_data_dir.glob('*')]))
    print(class_names)  
    no_of_classes = len(class_names)

    print('Number of Test Data Images')
    print(tf.data.experimental.cardinality(test_ds).numpy())

    test_ds = test_ds.map(lambda x: preprocess_classifier_test_images(x, class_names), 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_ds = configure_for_performance(test_ds, 1162, 
    tf.data.experimental.cardinality(test_ds).numpy())

    print('Retrieving Original Labels')
    original_labels =  np.load(cyclegan_predicted_labels_path)
    print(original_labels.shape)
    original_labels_ds = tf.data.Dataset.from_tensor_slices((original_labels))
    
   
    # Load of predicted Document Images
    generated_target_domain_images = np.load(cyclegan_predicted_images_path)
    print('Loaded target domain images shape')
    print(generated_target_domain_images.shape)

    generated_target_domain_images_ds = tf.data.Dataset.from_tensor_slices(
      (generated_target_domain_images))
    #print(tf.data.experimental.cardinality(generated_target_domain_images_ds).numpy())


    final_train_ds = tf.data.Dataset.zip((generated_target_domain_images_ds, 
    original_labels_ds))  
    print(tf.data.experimental.cardinality(final_train_ds).numpy())

    # Augment Training Data
    #final_train_ds = final_train_ds.map(preprocess_numpy_array_image, 
    #num_parallel_calls=tf.data.experimental.AUTOTUNE)

    X_train, y_train = next(iter(final_train_ds))

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train, cmap=plt.cm.gray)
    plt.show()
    plt.savefig(path + 'Loaded_Numpy_Sample_Augmented_Image_' + time)
    plt.close()

    plt.figure(figsize=(10,10))
    for i in range(10):
      plt.subplot(5, 2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(generated_target_domain_images[i], cmap=plt.cm.gray)
      plt.xlabel(class_names[np.argmax(original_labels[i], axis=-1)])
      plt.show()
      plt.savefig(path + 'Loaded_Numpy_Sample_Images_' + time)
    plt.close()

    # Train the Domain Adapted Realistic Document Image Classifier and 
    # Verify on Annotated Test Data.
    
    classifier_model = create_model(10)
    # Tensorboard Logs
    log_dir = path + "logs/fit/" + time   
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Creates a file writer for the log directory.
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
      tf.summary.image("10 training data examples", generated_target_domain_images[0:10], 
      max_outputs=25, step=0)
 
    # Prepare the plot
    figure = image_grid(generated_target_domain_images[0:10], class_names, 
    np.argmax(original_labels[0:10], axis=-1))

    # Convert to image and log
    with file_writer.as_default():
      tf.summary.image("Training data", plot_to_image(figure), step=0)

    # retrieve the test data
    X_test, y_test = next(iter(test_ds))

    # Shuffle final training dataset
    final_train_ds = final_train_ds.shuffle(tf.data.experimental.cardinality(final_train_ds).numpy())
    image_count = generated_target_domain_images.shape[0]

    val_size = int(image_count * 0.2)
    final_train_ds = final_train_ds.skip(val_size)
    val_ds = final_train_ds.take(val_size)

    final_train_ds = configure_for_performance(final_train_ds, 1,
    tf.data.experimental.cardinality(final_train_ds).numpy())
    val_ds = configure_for_performance(val_ds, 1,
    tf.data.experimental.cardinality(val_ds).numpy())

    file_name = path + 'Confusion_Matrix_' + classifier_name + '_' + time
    file_name_plot = classifier_name + '_' + time
    # Example: Confusion_Matrix_CycleGAN_Generated_Data_Classifier

    epochs = 20
    history = classifier_model.fit(
            final_train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[tensorboard_callback,
            CustomCallback(classifier_model, 
            X_test,
            np.argmax(y_test, axis=-1),
            class_names,
            log_dir,
            file_writer_cm,
            file_name
            )]
            ) 
    
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy', fontweight='bold', fontsize='x-large')
    plt.ylabel('Accuracy', fontweight='bold', fontsize='x-large')
    plt.xlabel('Epochs', fontweight='bold', fontsize='x-large')
    plt.legend(['Train', 'Validation'], loc='upper left', prop=dict(weight='bold'))
    plt.grid(True)
    plt.show()
    plt.savefig(path + file_name_plot + '_Accuracy', dpi=300)
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss', fontweight='bold', fontsize='x-large')
    plt.ylabel('Loss', fontweight='bold', fontsize='x-large')
    plt.xlabel('Epochs', fontweight='bold', fontsize='x-large')
    plt.legend(['Train', 'Validation'], loc='upper left', prop=dict(weight='bold'))
    plt.grid(True)
    plt.show()
    plt.savefig(path + file_name_plot + '_Loss', dpi=300)
    plt.close()

    print('Training Finished...')
    # serialize weights to HDF5
    
    classifier_model.save(path + classifier_name + '_' + time + '_model.h5')
    
    classifier_log_file =  open(path + classifier_name + '_logs'
    '_' + time + '.txt', 'a')
    
    print('Saved model to disk ' + classifier_name + '_' + time + '_model.h5', 
    file=classifier_log_file)


    print('translated target domain images and labels shape', file=classifier_log_file)
    print(generated_target_domain_images.shape, file=classifier_log_file) 
    print(original_labels.shape, file=classifier_log_file)


    y_test_pred = np.argmax(classifier_model.predict(X_test), axis=-1)
    y_test_real = np.argmax(y_test, axis=-1)

    results = classifier_model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_real, y_test_pred, target_names=class_names, zero_division=1), 
    file=classifier_log_file)
    
    print("test loss, test acc:", results, file=classifier_log_file)
    classifier_log_file.close()