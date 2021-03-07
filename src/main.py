from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from cyclegan_resnet import *
from datetime import datetime
import sys
import tensorflow as tf
from preprocessing_images import preprocess_cyclegan_images
import pathlib
from preprocessing_images import *

# Algorithm Steps:
# Train the CycleGAN.
# Train the Synthetic Document Image Classifier and Verify on Annotated Test Data.


if __name__ == "__main__":


    print(f"Arguments count: {len(sys.argv)}")
    synthetic_document_images_path = sys.argv[1]
    real_document_images_path = sys.argv[2]
    tf.debugging.set_log_device_placement(True)
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


    ds_source_domain_list_files = tf.data.Dataset.list_files(str(pathlib.Path(synthetic_document_images_path+'*.png')))
    ds_source_domain_dataset = ds_source_domain_list_files.map(preprocess_cyclegan_images,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_source_domain_dataset = configure_for_performance(ds_source_domain_dataset, 1)   

    ds_target_domain_list_files = tf.data.Dataset.list_files(str(pathlib.Path(real_document_images_path+'*.png')))
    ds_target_domain_dataset = ds_target_domain_list_files.map(preprocess_cyclegan_images,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


    ds_target_domain_dataset = configure_for_performance(ds_target_domain_dataset, 1) 

    train_cyclegan(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, 
        c_model_AtoB, c_model_BtoA, ds_source_domain_dataset, ds_target_domain_dataset, cyclegan_training_logs)

    cyclegan_training_logs.close()
    





