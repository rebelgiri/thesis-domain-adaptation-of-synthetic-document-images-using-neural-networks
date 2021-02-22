from keras.models import load_model
from sklearn.utils import validation
from tensorflow.python.ops.gen_batch_ops import batch
from synthetic_documents_classifier import *
import sys
from datetime import datetime
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# from data_set_loader_pytorch import *
from data_set_loader_keras import *
import numpy as np


# Algorithm Steps:
# Load the CycleGAN Model Generate the Domain Adapted Realistic Document Images.
# Train the Domain Adapted Realistic Document Image Classifier and Verify in Annotated Test Data.


if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    cyclegan_generator_path = sys.argv[1]
    classifier_training_data_set_path = sys.argv[2]
    classifier_test_data_set_path = sys.argv[3]

    # create generator
    datagen = ImageDataGenerator(preprocessing_function=normalize)

    # prepare an iterators for training dataset
    train_it = datagen.flow_from_directory(
        classifier_training_data_set_path, 
        color_mode='grayscale',
        shuffle=True,
        batch_size=10,
        interpolation='bilinear',
        subset='training')

    # prepare an iterators for testing dataset
    test_it = datagen.flow_from_directory(
        classifier_test_data_set_path, 
        color_mode='grayscale',
        shuffle=True,
        batch_size=1162,
        interpolation='bilinear')    

    classes = list()
    for key in train_it.class_indices:
        classes.append(key)

     # Load the CycleGAN Generator Model to generate the Domain Adapted Document Images.
    model = load_model(cyclegan_generator_path,
    custom_objects={'InstanceNormalization': InstanceNormalization})


    total_no_images = train_it.n  
    steps = int(total_no_images/ 10) # 10000
    translated_target_domain_images = np.array([])
    source_domain_labels = np.array([])


    for i in range(steps): #steps
        source_domain_images_batch , source_domain_labels_batch = train_it.next()

        source_domain_labels = np.vstack([source_domain_labels, 
        source_domain_labels_batch]) if source_domain_labels.size else source_domain_labels_batch

        # generate images
        translated_target_domain_pred_images = model.predict(source_domain_images_batch)
        translated_target_domain_images = np.vstack([translated_target_domain_images, 
        translated_target_domain_pred_images]) if translated_target_domain_images.size else translated_target_domain_pred_images


    print('translated target domain images and labels shape')
    print(translated_target_domain_images.shape) 
    print(source_domain_labels.shape)

    # Train the Domain Adapted Realistic Document Image Classifier and 
    # Verify on Annotated Test Data.
   
    type_of_the_classifier = 'domain_adapted_documents_classifier'
    domain_adapted_documents_classifier_model = create_model(10)

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    domain_adapted_documents_classifier_logs =  open('domain_adapted_documents_classifier_logs'
    '_' + time + '.txt', 'a')
    log_dir = "logs/fit/" + time
    
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = list()
    
    domain_adapted_documents_classifier_model.fit(
            translated_target_domain_images, 
            source_domain_labels,
            epochs=10,
            batch_size=10,
            validation_split=0.2)
    # callbacks=[tensorboard_callback]))      
    print('Training Finished...')

   

    print('translated target domain images and labels shape', file=domain_adapted_documents_classifier_logs)
    print(translated_target_domain_images.shape, file=domain_adapted_documents_classifier_logs) 
    print(source_domain_labels.shape, file=domain_adapted_documents_classifier_logs)

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


    X_test, y_test = test_it.next()

    y_test_pred = np.argmax(domain_adapted_documents_classifier_model.predict(X_test), axis=-1)
    y_test_real = np.argmax(y_test, axis=-1)


    results = domain_adapted_documents_classifier_model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_real, y_test_pred, target_names=classes, zero_division=1), 
    file=domain_adapted_documents_classifier_logs)
    
    print("test loss, test acc:", results, file=domain_adapted_documents_classifier_logs)

    # serialize weights to HDF5
    domain_adapted_documents_classifier_model.save(type_of_the_classifier + '_' + time + '_model.h5')

    print('Saved model to disk ' + type_of_the_classifier + '_' + time + '_model.h5', 
    file=domain_adapted_documents_classifier_logs)

    domain_adapted_documents_classifier_logs.close()

    