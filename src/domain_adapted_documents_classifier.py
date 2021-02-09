from keras.models import load_model
from synthetic_documents_classifier import *
import sys
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from sklearn.utils import shuffle


# Algorithm Steps:
# Load the CycleGAN Model Generate the Domain Adapted Realistic Document Images.
# Train the Domain Adapted Realistic Document Image Classifier and Verify in Annotated Test Data.


if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    cyclegan_generator_path = sys.argv[1]
    classifier_training_data_set_path = sys.argv[2]
    classifier_test_data_set_path = sys.argv[3]

    
    domain_adapted_documents_classifier_logs =  open('domain_adapted_documents_classifier_logs.txt', 'a')
    print(datetime.now(), file=domain_adapted_documents_classifier_logs)
    
    # Load the CycleGAN Generator Model to generate the Domain Adapted Document Images.
    model = load_model(cyclegan_generator_path,
    custom_objects={'InstanceNormalization': InstanceNormalization})

    source_domain_data_set, list_of_name_of_template = classifier_load_data_set(classifier_training_data_set_path)


    source_domain_images, source_domain_labels = source_domain_data_set

    # generate images
    translated_target_domain_images = model.predict(source_domain_images)

    # shuffle data set
    translated_target_domain_images, source_domain_labels = shuffle(translated_target_domain_images, source_domain_labels)

    # Train the Domain Adapted Realistic Document Image Classifier and 
    # Verify on Annotated Test Data.
   
    classifier_test_data_set, list_of_name_of_template = classifier_load_data_set(classifier_test_data_set_path)
    type_of_the_classifier = 'domain_adapted_documents_classifier'
    domain_adapted_documents_classifier_model = create_model(10)

    start_training_classifier(domain_adapted_documents_classifier_model, 
        (translated_target_domain_images, source_domain_labels), 
        classifier_test_data_set, type_of_the_classifier, list_of_name_of_template, domain_adapted_documents_classifier_logs)

    print('---------------------------------------------------------------------------------', 
    file=domain_adapted_documents_classifier_logs)

    domain_adapted_documents_classifier_logs.close()        
