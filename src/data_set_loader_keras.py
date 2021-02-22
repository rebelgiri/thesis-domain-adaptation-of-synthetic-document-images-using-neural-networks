
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import sys

def normalize(image):
  image = (image / 127.5) - 1
  return image

def  classifier_data_set_loader_keras(classifier_training_data_set_path, 
classifier_test_data_set_path):


    # create generator
    datagen = ImageDataGenerator(preprocessing_function=normalize())
    # prepare an iterators for each dataset
    train_it = datagen.flow_from_directory(classifier_training_data_set_path, 
    color_mode='grayscale',
    shuffle=True,
    batch_size=10,
    interpolation='bilinear')

    test_it = datagen.flow_from_directory(classifier_test_data_set_path, 
    color_mode='grayscale',
    shuffle=True,
    batch_size=1162,
    interpolation='bilinear')

    
    datagen.standardize()
    # confirm the iterator works
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))





if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    classifier_training_data_set_path = sys.argv[1]
    classifier_test_data_set_path = sys.argv[2]
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



    classifier_data_set_loader_keras(
        classifier_training_data_set_path, classifier_test_data_set_path)