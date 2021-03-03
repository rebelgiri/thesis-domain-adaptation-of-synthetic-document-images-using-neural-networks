# Reference 
# https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
# https://www.tensorflow.org/tutorials/generative/cyclegan
# pip install git+https://www.github.com/keras-team/keras-contrib.git


from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers import RandomNormal
from keras.layers import Conv2DTranspose
from numpy.random import randint
from numpy import zeros
from numpy import ones
from numpy import asarray
from matplotlib import pyplot
from random import random

# example of defining a 70x70 patchgan discriminator model
# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5], metrics=['accuracy'])
    return model

# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g

# example of an encoder-decoder generator for the cyclegan
# define the standalone generator model
def define_generator(image_shape=(256, 256, 1), n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(1, (7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    # identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # forward cycle
    output_f = g_model_2(gen1_out)
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    # define model graph
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    # define optimization algorithm configuration
    opt = Adam(lr=0.0002, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

# train cyclegan models
def train_cyclegan(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, 
    ds_source_domain_dataset, ds_target_domain_dataset, cyclegan_training_logs):

    # define properties of the training run
    n_epochs, n_batch = 1, 10

    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(100000 / n_batch)
        
    # manually enumerate epochs
    for i in range(n_epochs):
        ds_source_domain_dataset_iter = iter(ds_source_domain_dataset)
        ds_target_domain_dataset_iter = iter(ds_target_domain_dataset)

        for j in range(1): # bat_per_epo
            trainA, _ = next(ds_source_domain_dataset_iter) # Source Domain Images
            trainB, _ = next(ds_target_domain_dataset_iter) # Target Domain Images
            
            trainA = asarray(trainA)
            trainB = asarray(trainB)
            
            # select a batch of real samples
            X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
            X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
            # update fakes from pool
            X_fakeA = update_image_pool(poolA, X_fakeA)
            X_fakeB = update_image_pool(poolB, X_fakeB)
            # update generator B->A via adversarial and cycle loss
            g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            # update discriminator for A -> [real/fake]
            dA_loss1, _ = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2, _ = d_model_A.train_on_batch(X_fakeA, y_fakeA)
            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            # update discriminator for B -> [real/fake]
            dB_loss1, _ = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2, _ = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            
            # summarize performance
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (
                j + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (
                j + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2), file=cyclegan_training_logs)

        # evaluate the model performance, sometimes
        # if (j + 1) % 10000 == 0:
        
        summarize_performance(i, g_model_AtoB, 'g_model_AtoB', d_model_B, 'd_model_B', 
                (trainA, trainB), n_batch, n_patch, cyclegan_training_logs)

        summarize_performance(i, g_model_BtoA, 'g_model_BtoA', d_model_A, 'd_model_A', 
                (trainB, trainA), n_batch, n_patch, cyclegan_training_logs)

            
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, g_model_name, d_model, d_model_name, dataset, 
    n_batch, n_patch, cyclegan_training_logs):

    # unpack dataset
    trainS, trainT = dataset

    # select a batch of real samples
    X_realS, _ = generate_real_samples(trainS, n_batch, n_patch)
    X_realT, y_realT = generate_real_samples(trainT, n_batch, n_patch)
    
    # generate a batch of fake samples
    X_fakeT, y_fakeT = generate_fake_samples(g_model, X_realS, n_patch)

    # evaluate discriminator on real examples
    _, acc_realT = d_model.evaluate(X_realT, y_realT)

    # evaluate discriminator on fake examples
    _, acc_fakeT = d_model.evaluate(X_fakeT, y_fakeT)

    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_realT * 100, acc_fakeT * 100), 
    file=cyclegan_training_logs)

    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_realT * 100, acc_fakeT * 100))
    
    # save plot
  
    save_plot(X_fakeT, epoch, n_batch, g_model_name)
    # save the generator model tile file
    filename_g_model = 'generator_model_%s_%03d.h5' % (g_model_name, epoch + 1)
    filename_d_model = 'discriminator_model_%s_%03d.h5' % (d_model_name, epoch + 1)
    g_model.save(filename_g_model)
    d_model.save(filename_d_model)

# create and save a plot of generated images
def save_plot(examples, epoch, n, generator_model_name):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        image = examples[i]
        # we need to get rid of the last dimension
        image = image[:, :, 0]
        pyplot.imshow(image, cmap='gray')
    # save plot to file
    filename = 'generated_plot_%s_e%03d.png' % (generator_model_name, epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()



    

