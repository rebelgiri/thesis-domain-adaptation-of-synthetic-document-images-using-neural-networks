
from synthetic_documents_classifier import *
from datetime import datetime
from data_set_loader_pytorch import *
import sys
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    classifier_training_data_set_path = sys.argv[1]
    classifier_test_data_set_path = sys.argv[2]
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    datagen = ImageDataGenerator()
    

    exit(0)

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Train the Synthetic Document Image Classifier and Verify in Annotated Test Data.
    
    type_of_the_classifier = 'synthetic_documents_classifier'
    synthetic_documents_classifier_model = create_model(10)

    (classifier_training_images_data_set_loader, classifier_test_images_data_set_loader, classes) = classifier_data_set_loader(
        classifier_training_data_set_path, classifier_test_data_set_path)


    synthetic_documents_classifier_logs =  open('synthetic_documents_classifier_logs' + 
    '_' + time + '.txt', 'a')
    
    start_training_synthetic_documents_classifier(synthetic_documents_classifier_model, 
        classifier_training_images_data_set_loader, classifier_test_images_data_set_loader, type_of_the_classifier,
        classes, synthetic_documents_classifier_logs, time)

    print('---------------------------------------------------------------------------------', 
    file=synthetic_documents_classifier_logs)

    
    synthetic_documents_classifier_logs.close()