from keras.models import load_model
from synthetic_documents_classifier import *
import sys
from datetime import datetime
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from sklearn.utils import shuffle
from data_loader import *
import numpy as np


# Algorithm Steps:
# Load the CycleGAN Model Generate the Domain Adapted Realistic Document Images.
# Train the Domain Adapted Realistic Document Image Classifier and Verify in Annotated Test Data.


def start_training_domain_adapted_documents_classifier(model, 
        dataset, 
        classifier_test_images_data_set_iter, type_of_the_classifier, classes, 
        domain_adapted_documents_classifier_logs):

    X_train, y_train = dataset
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Pre-processing class labels
    y_train = np_utils.to_categorical(y_train, 10)
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=100,  validation_split=0.2,
        callbacks=[tensorboard_callback])


    # Save the results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Error in percentage')
    plt.legend(loc='lower right')
    plt.grid('on')
    plt.show()
    plt.savefig(type_of_the_classifier + '_' + time + '_model.png')
    plt.close()
    
    # Evaluate on real annoted data
    X_test_tensor, y_test_tensor = classifier_test_images_data_set_iter.next()
    X_test = X_test_tensor.numpy()
    X_test = np.einsum('ijkl->iklj', X_test)
    y_test = y_test_tensor.numpy()
    y_test_true = y_test

    # Pre-processing class labels
    y_test = np_utils.to_categorical(y_test, 10)

    print('Evaluation Results', file=domain_adapted_documents_classifier_logs)
    print(X_test.shape, file=domain_adapted_documents_classifier_logs)
    print(y_test.shape, file=domain_adapted_documents_classifier_logs)

    y_test_pred = np.argmax(model.predict(X_test), axis=-1)
    
    results = model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_true, y_test_pred, target_names=classes, zero_division=1), 
    file=domain_adapted_documents_classifier_logs)
    print("test loss, test acc:", results, file=domain_adapted_documents_classifier_logs)

    # serialize weights to HDF5
    model.save(type_of_the_classifier + '_' + time + '_model.h5')
    
    print('Saved model to disk ' + type_of_the_classifier,
    file=domain_adapted_documents_classifier_logs)

if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    cyclegan_generator_path = sys.argv[1]
    classifier_training_data_set_path = sys.argv[2]
    classifier_test_data_set_path = sys.argv[3]

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    domain_adapted_documents_classifier_logs =  open('domain_adapted_documents_classifier_logs'
    '_' + time + '_.txt', 'a')
    
    # Load the CycleGAN Generator Model to generate the Domain Adapted Document Images.
    model = load_model(cyclegan_generator_path,
    custom_objects={'InstanceNormalization': InstanceNormalization})

    # source_domain_data_set, list_of_name_of_template = classifier_load_data_set(
    # classifier_training_data_set_path)
    # source_domain_images, source_domain_labels = source_domain_data_set
    # generate images
    # translated_target_domain_images = model.predict(source_domain_images)
    # shuffle data set
    # translated_target_domain_images, source_domain_labels = shuffle(translated_target_domain_images, source_domain_labels)

    translated_target_domain_images = np.array([])
    source_domain_labels = np.array([])
    (classifier_training_images_data_set_iter, 
    classifier_test_images_data_set_iter, classes) = classifier_data_set_loader(
        classifier_training_data_set_path, classifier_test_data_set_path)

    n_batch = 1000
    for i in range(n_batch):
        (source_domain_images_tensor, 
        source_domain_labels_tensor) = classifier_training_images_data_set_iter.next()
        source_domain_images_tensor_numpy = source_domain_images_tensor.numpy()
        source_domain_images = np.einsum('ijkl->iklj', source_domain_images_tensor_numpy)
        source_domain_labels_numpy = source_domain_labels_tensor.numpy()
        
        source_domain_labels = np.vstack([source_domain_labels, 
        source_domain_labels_numpy]) if source_domain_labels.size else source_domain_labels_numpy

        # generate images
        translated_target_domain_pred_images = model.predict(source_domain_images)
        translated_target_domain_images = np.vstack([translated_target_domain_images, 
        translated_target_domain_pred_images]) if translated_target_domain_images.size else translated_target_domain_pred_images

    print('translated target domain images shape', file=domain_adapted_documents_classifier_logs)
    print(translated_target_domain_images.shape, file=domain_adapted_documents_classifier_logs) 
    print(source_domain_labels.shape, file=domain_adapted_documents_classifier_logs)

    # Train the Domain Adapted Realistic Document Image Classifier and 
    # Verify on Annotated Test Data.
   
    type_of_the_classifier = 'domain_adapted_documents_classifier'
    domain_adapted_documents_classifier_model = create_model(10)

    start_training_domain_adapted_documents_classifier(domain_adapted_documents_classifier_model, 
        (translated_target_domain_images, source_domain_labels), 
        classifier_test_images_data_set_iter, type_of_the_classifier, classes, 
        domain_adapted_documents_classifier_logs)

    print('---------------------------------------------------------------------------------', 
    file=domain_adapted_documents_classifier_logs)
    domain_adapted_documents_classifier_logs.close()        



    