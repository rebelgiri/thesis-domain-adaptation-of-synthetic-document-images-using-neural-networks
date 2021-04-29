"""
Title: CycleGAN
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2020/08/12
Last modified: 2020/08/12
Description: Implementation of CycleGAN.
"""

"""
## CycleGAN
CycleGAN is a model that aims to solve the image-to-image translation
problem. The goal of the image-to-image translation problem is to learn the
mapping between an input image and an output image using a training set of
aligned image pairs. However, obtaining paired examples isn't always feasible.
CycleGAN tries to learn this mapping without requiring paired input-output images,
using cycle-consistent adversarial networks.
- [Paper](https://arxiv.org/pdf/1703.10593.pdf)
- [Original implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
"""

"""
## Setup
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

"""
## Prepare the dataset
"""
#synthetic_document_images_path = '/mnt/data/data/cyclegan_synthetic_document_images/synthetic_document_images/'
#real_document_images_path = '/mnt/data/data/cyclegan_real_document_images/real_document_images/'

synthetic_document_images_path = '/mnt/data/data/cyclegan_synthetic_document_images_test/'
real_document_images_path = '/mnt/data/data/cyclegan_real_document_images_test/'
synthetic_document_images_path_test = '/mnt/data/data/cyclegan_synthetic_document_images_test/'


# Define the standard image size.
input_img_size = (256, 256, 1)

# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = 256
batch_size = 1


def normalize_img(img):
    # Map values in the range [-1, 1]
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def preprocess_cyclegan_images(image_path):
    img = tf.io.read_file(image_path)
    # Convert the image in grayscale
    img = tf.image.decode_png(img, channels=1)
    # Resize the image [[256, 256]]
    img = tf.image.resize(img, [256, 256])
    # Map values in the range [-1, 1]
    img = normalize_img(img)
    return img



# Training Dataset
ds_source_domain_list_files = tf.data.Dataset.list_files(str(pathlib.Path(synthetic_document_images_path+'*.png')))
ds_source_domain_dataset = ds_source_domain_list_files.map(preprocess_cyclegan_images,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)

ds_target_domain_list_files = tf.data.Dataset.list_files(str(pathlib.Path(real_document_images_path+'*.png')))
ds_target_domain_dataset = ds_target_domain_list_files.map(preprocess_cyclegan_images,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)


# Testing Dataset
ds_source_domain_list_files_test = tf.data.Dataset.list_files(str(
    pathlib.Path(synthetic_document_images_path_test+'*.png')))

ds_source_domain_dataset_test = ds_source_domain_list_files_test.map(preprocess_cyclegan_images,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)




# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './cyclegan_parallel_programming/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


"""
## Create `Dataset` objects
"""
BATCH_SIZE_PER_REPLICA = batch_size
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  

"""
## Visualize some samples
"""

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(ds_source_domain_dataset.take(4), ds_target_domain_dataset.take(4))):
    source = ((samples[0][0]).numpy())
    target = ((samples[1][0]).numpy())
    ax[i, 0].imshow(source, cmap=plt.cm.gray)
    ax[i, 1].imshow(target, cmap=plt.cm.gray)
plt.show()
plt.savefig('Visualize_Some_Samples')   


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


'''
c7s1-64 ==> Conv block with `relu` activation, filter size of 7
d128 ====|
         |-> 2 downsampling blocks
d256 ====|
R256 ====|
R256     |
R256     |
R256     |
R256     |-> 9 residual blocks
R256     |
R256     |
R256     |
R256 ====|
u128 ====|
         |-> 2 upsampling blocks
u64  ====|
c7s1-3 => Last conv block with `tanh` activation, filter size of 7.
'''

def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(1, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model



def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        total_loss = total_loss_G + total_loss_F + disc_X_loss + disc_Y_loss

        return total_loss
        '''
        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }
        '''


# Open a strategy scope.
with strategy.scope():

    mae_loss_fn = keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    
    # Loss function for evaluating cycle consistency loss
    def cycle_loss_fn(real, cycled):
        cycle_loss = mae_loss_fn(real, cycled)
        cycle_loss = tf.nn.compute_average_loss(cycle_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return cycle_loss
         
    # Loss function for evaluating identity mapping loss
    def identity_loss_fn(real, same):
        identity_loss = mae_loss_fn(real, same)
        identity_loss = tf.nn.compute_average_loss(identity_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return identity_loss

    # Loss function for evaluating adversarial loss
    adv_loss_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    # Define the loss function for the generators
    def generator_loss_fn(fake):
        fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
        fake_loss = tf.nn.compute_average_loss(fake_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return fake_loss

    # Define the loss function for the discriminators
    def discriminator_loss_fn(real, fake):
        real_loss = adv_loss_fn(tf.ones_like(real), real)
        fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
        real_loss = tf.nn.compute_average_loss(real_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        fake_loss = tf.nn.compute_average_loss(fake_loss, global_batch_size=GLOBAL_BATCH_SIZE)  
        return (real_loss + fake_loss) * 0.5

    # Get the generators
    gen_G = get_resnet_generator(name="generator_G")
    gen_F = get_resnet_generator(name="generator_F")

    # Get the discriminators
    disc_X = get_discriminator(name="discriminator_X")
    disc_Y = get_discriminator(name="discriminator_Y")

    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)   

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=optimizer,
        gen_F_optimizer=optimizer,
        disc_X_optimizer=optimizer,
        disc_Y_optimizer=optimizer,
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
        cycle_loss_fn=cycle_loss_fn,
        identity_loss_fn=identity_loss_fn
    )


    checkpoint = tf.train.Checkpoint(
                            gen_G=gen_G,
                            gen_F=gen_F,
                            disc_X=disc_X,
                            disc_Y=disc_Y,
                            gen_G_optimizer=optimizer,
                            gen_F_optimizer=optimizer,
                            disc_X_optimizer=optimizer,
                            disc_Y_optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=5)

    train_dist_dataset = strategy.experimental_distribute_dataset(
        tf.data.Dataset.zip((ds_source_domain_dataset, ds_target_domain_dataset)))

    
# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(cycle_gan_model.train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

for epoch in range(1):
    # TRAIN LOOP
    all_loss = 0.0
    num_batches = 0.0
    for one_batch in train_dist_dataset:
        all_loss +=  distributed_train_step(one_batch)
        num_batches += 1
    
    train_loss = all_loss/num_batches
    checkpoint.save(checkpoint_prefix)
    print(train_loss)

        
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  checkpoint.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
      
