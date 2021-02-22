
from cyclegan_resnet import *
from synthetic_documents_classifier import *
from datetime import datetime
from data_set_loader_pytorch import *
import sys

# Algorithm Steps:
# Train the CycleGAN.
# Train the Synthetic Document Image Classifier and Verify on Annotated Test Data.


if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    synthetic_document_images_path = sys.argv[1]
    real_document_images_path = sys.argv[2]
    classifier_training_data_set_path = sys.argv[3]
    classifier_test_data_set_path = sys.argv[4]

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cyclegan_training_logs = open('cyclegan_training_logs' + '_' + time + '_.txt', 'a')

    # Train the CycleGAN.
    
    # input shape
    image_shape = (256, 256, 1)

    # generator: A -> B
    g_model_AtoB = define_generator(image_shape)
   
    # generator: B -> A
    g_model_BtoA = define_generator(image_shape)
   
    # discriminator: A -> [real/fake]
    d_model_A = define_discriminator(image_shape)
   
    # discriminator: B -> [real/fake]
    d_model_B = define_discriminator(image_shape)
   
    # composite: A -> B -> [real/fake, A]
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    
    # composite: B -> A -> [real/fake, B]
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
  

    data_set_cyclegan_loader = cyclegan_data_set_loader(synthetic_document_images_path, real_document_images_path)    
    
   
    train_cyclegan(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, 
        c_model_AtoB, c_model_BtoA, data_set_cyclegan_loader, cyclegan_training_logs)

    cyclegan_training_logs.close()    

    # Train the Synthetic Document Image Classifier and Verify in Annotated Test Data.
    type_of_the_classifier = 'synthetic_documents_classifier'
    synthetic_documents_classifier_model = create_model(10)
    
    (classifier_training_images_data_set_loader, classifier_test_images_data_set_loader, classes) = classifier_data_set_loader(
        classifier_training_data_set_path, classifier_test_data_set_path)


    synthetic_documents_classifier_logs =  open('synthetic_documents_classifier_logs' + 
    '_' + time + '.txt', 'a')
    
    start_training_synthetic_documents_classifier(synthetic_documents_classifier_model, 
        classifier_training_images_data_set_loader, classifier_test_images_data_set_loader, type_of_the_classifier,
        classes, synthetic_documents_classifier_logs)

    synthetic_documents_classifier_logs.close()
    





