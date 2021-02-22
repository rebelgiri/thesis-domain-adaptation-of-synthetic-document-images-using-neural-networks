# example of loading the generator model and generating images
from keras.models import load_model
from matplotlib import pyplot
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from data_set_loader_pytorch import *
import numpy as np

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
model = load_model('/home/giriraj/thesis-domain-adaptation-of-synthetic-generated-documents/thesis-domain-adaptation-of-synthetic-generated-documents/generator_model_g_model_AtoB_005.h5',
custom_objects={'InstanceNormalization': InstanceNormalization})


test_images_folder = torchvision.datasets.ImageFolder(
        root='/home/giriraj/data/synthetic_document_images_test',
        transform=transformations['classifier']
    )

test_images_loader = torch.utils.data.DataLoader(
        test_images_folder,
        batch_size=55,
        num_workers=4,
        shuffle=True
    )


test_images_iter = iter(test_images_loader)
test_data_set, _ = test_images_iter.next() # Test Synthetic Domain Images
test_data_set = test_data_set.numpy()
test_data_set = np.einsum('ijkl->iklj', test_data_set)


print(test_data_set.shape)
# generate images
X = model.predict(test_data_set)

# plot the result
save_plot(test_data_set, X, 5)