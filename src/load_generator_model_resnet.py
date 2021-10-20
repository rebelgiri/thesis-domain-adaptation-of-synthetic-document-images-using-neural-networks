# example of loading the generator model and generating images
from keras.models import load_model
from matplotlib import pyplot
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import ImageDataGenerator
from train_synthetic_documents_classfier import normalize


# create and save a plot of generated images (reversed grayscale)
def save_plot(test_data_set, examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(1, 2, 1)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(test_data_set[i], cmap='gray')

        pyplot.subplot(1, 2, 2)
         # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i], cmap='gray')

        pyplot.savefig('Results_From_ResNet_Generator_%d.png' % (i))
        pyplot.close()


# load model    
model = load_model('/home/giriraj/cyclegan_models/generator_model_g_model_AtoB_003.h5',
custom_objects={'InstanceNormalization': InstanceNormalization})


# create generator
datagen = ImageDataGenerator(preprocessing_function=normalize)

# prepare an iterators for training dataset
test_it = datagen.flow_from_directory('/mnt/data/data/synthetic_document_images_test/', 
    color_mode='grayscale',
    shuffle=True,
    batch_size=54,
    interpolation='bilinear',
    subset='training')

test_data_set, _ = test_it.next()

print(test_data_set.shape)

# generate images
X = model.predict(test_data_set)

# plot the result
save_plot(test_data_set, X, 5)