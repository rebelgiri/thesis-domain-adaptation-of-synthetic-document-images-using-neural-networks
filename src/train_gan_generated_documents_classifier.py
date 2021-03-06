from keras.models import load_model
from sklearn.model_selection import train_test_split
from classifier_model import *
from train_synthetic_documents_classifier import *
import sys
from datetime import datetime
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# Algorithm Steps:
# Load the CycleGAN Model Generate the Domain Adapted Realistic Document Images.
# Train the Domain Adapted Realistic Document Image Classifier and Verify in Annotated Test Data.


if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    cyclegan_generator_path = sys.argv[1]
    classifier_training_data_set_path = sys.argv[2]
    classifier_test_data_set_path = sys.argv[3]

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    data_dir = pathlib.Path(classifier_training_data_set_path)
    train_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

    test_data_dir = pathlib.Path(classifier_test_data_set_path)
    test_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*'), shuffle=False)

    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    print(class_names)  
    no_of_classes = len(class_names)

    list_of_files = list(data_dir.glob('*/*.png'))
    print(list(data_dir.glob('*/*.png')))  

    image_count = len(list(data_dir.glob('*/*.png')))
    print(image_count)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())

    train_ds = train_ds.map(preprocess_classifier_images, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(preprocess_classifier_images, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = configure_for_performance_without_shuffle(train_ds, 1)
    test_ds = configure_for_performance(test_ds, 1162)

    print('Retrieving Original Labels')
    original_labels =  np.asarray(list(train_ds.map(lambda y, x: x)))
    
    print(original_labels.shape)
    original_labels = np.reshape(original_labels, (image_count, no_of_classes))
    print(original_labels.shape)
    original_labels_ds = tf.data.Dataset.from_tensor_slices((original_labels))

 
   
    plt.figure(figsize=(10,10))
    for i in range(10):
      X_train_10_samples, y_train_10_samples = next(iter(train_ds))
      plt.subplot(5, 2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(X_train_10_samples[0], cmap=plt.cm.gray)
      plt.xlabel(class_names[np.argmax(y_train_10_samples[0], axis=-1)])
      plt.show()
      plt.savefig('Save_Sample_Images_1')
   

   
    plt.figure(figsize=(10,10))
    for i in range(10):
      X_train_10_samples, y_train_10_samples = next(iter(train_ds))
      plt.subplot(5, 2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(X_train_10_samples[0], cmap=plt.cm.gray)
      plt.xlabel(class_names[np.argmax(y_train_10_samples[0], axis=-1)])
      plt.show()
      plt.savefig('Save_Sample_Images_2')
   
 
 
    # Load the CycleGAN Generator Model to generate the Domain Adapted Document Images.
    model = load_model(cyclegan_generator_path,
    custom_objects={'InstanceNormalization': InstanceNormalization})

    # Generation of Target Domain Document Images
    generated_target_domain_images = model.predict(train_ds)
    print('Generated target domain images and labels shape')
    print(generated_target_domain_images.shape) 


    generated_target_domain_images_ds = tf.data.Dataset.from_tensor_slices((generated_target_domain_images))
    print(tf.data.experimental.cardinality(generated_target_domain_images_ds).numpy())

    final_train_ds =  tf.data.Dataset.zip((generated_target_domain_images_ds, original_labels_ds))  
    

    plt.figure(figsize=(10,10))
    for i in range(10):
      plt.subplot(5, 2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(generated_target_domain_images[i], cmap=plt.cm.gray)
      plt.xlabel(class_names[np.argmax(original_labels[i], axis=-1)])
      plt.show()
      plt.savefig('Generated_Sample_Images')
 
    
    # Train the Domain Adapted Realistic Document Image Classifier and 
    # Verify on Annotated Test Data.
   
    type_of_the_classifier = 'domain_adapted_documents_classifier'
    domain_adapted_documents_classifier_model = create_model(10)

    
 
    # Tensorboard Logs
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = "logs/fit/" + time   
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

    

    val_size = int(image_count * 0.2)
    final_train_ds = final_train_ds.skip(val_size)
    val_ds = final_train_ds.take(val_size)

    final_train_ds = configure_for_performance(final_train_ds, 10)
    val_ds = configure_for_performance(val_ds, 10)

    domain_adapted_documents_classifier_model.fit(
            final_train_ds,
            epochs=10,
            validation_data=val_ds,
            callbacks=[tensorboard_callback,
            CustomCallback(domain_adapted_documents_classifier_model, 
            X_test,
            np.argmax(y_test, axis=-1),
            class_names,
            log_dir,
            file_writer_cm
            )]
            ) 


    print('Training Finished...')
     # serialize weights to HDF5
    domain_adapted_documents_classifier_model.save(type_of_the_classifier + '_' + time + '_model.h5')
    
    domain_adapted_documents_classifier_logs =  open('domain_adapted_documents_classifier_logs'
    '_' + time + '.txt', 'a')
    
    print('Saved model to disk ' + type_of_the_classifier + '_' + time + '_model.h5', 
    file=domain_adapted_documents_classifier_logs)


    print('translated target domain images and labels shape', file=domain_adapted_documents_classifier_logs)
    print(generated_target_domain_images.shape, file=domain_adapted_documents_classifier_logs) 
    print(original_labels.shape, file=domain_adapted_documents_classifier_logs)


    y_test_pred = np.argmax(domain_adapted_documents_classifier_model.predict(X_test), axis=-1)
    y_test_real = np.argmax(y_test, axis=-1)

    results = domain_adapted_documents_classifier_model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_real, y_test_pred, target_names=class_names, zero_division=1), 
    file=domain_adapted_documents_classifier_logs)
    
    print("test loss, test acc:", results, file=domain_adapted_documents_classifier_logs)

  
    domain_adapted_documents_classifier_logs.close()

