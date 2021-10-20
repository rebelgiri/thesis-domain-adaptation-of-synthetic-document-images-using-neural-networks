from cyclegan_resnet import *
from datetime import datetime
import sys
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from train_synthetic_documents_classfier import normalize

# Algorithm Steps:
# Train the CycleGAN.
# Train the Synthetic Document Image Classifier and Verify on Annotated Test Data.


if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    synthetic_document_images_path = sys.argv[1]
    real_document_images_path = sys.argv[2]

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

    datagen = ImageDataGenerator(
        preprocessing_function=normalize)

    # prepare an iterators for each dataset
    source_domain_it = datagen.flow_from_directory(synthetic_document_images_path, 
    color_mode='grayscale',
    shuffle=True,
    batch_size=10,
    interpolation='bilinear')
  

    target_domain_it = datagen.flow_from_directory(real_document_images_path, 
    color_mode='grayscale',
    shuffle=True,
    batch_size=10,
    interpolation='bilinear')
    
    train_cyclegan(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, 
        c_model_AtoB, c_model_BtoA, source_domain_it, target_domain_it, cyclegan_training_logs)

    cyclegan_training_logs.close()
    





