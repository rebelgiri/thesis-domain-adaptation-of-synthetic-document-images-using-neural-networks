

from keras.models import load_model
from classifier_model import *
from train_synthetic_documents_classifier import *
import sys
from datetime import datetime
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE
from preprocessing_images import *
from numpy import load, save


buffer_size = 256
batch_size = 1

# Define the standard image size.
orig_img_size = (286, 286)
# Size of the random crops to be used during training.
input_img_size = (256, 256, 1)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

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


"""
## Build the generators
The generator consists of downsampling blocks: nine residual blocks
and upsampling blocks. The structure of the generator is the following:
```
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
```
"""


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


"""
## Build the discriminators
The discriminators implement the following architecture:
`C64->C128->C256->C512`
"""


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


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")


"""
## Build the CycleGAN model
We will override the `train_step()` method of the `Model` class
for training via `fit()`.
"""


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
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Source and y is Target
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
            # Source to fake Target
            fake_y = self.gen_G(real_x, training=True)
            # Target to fake Source -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Source to fake Target to fake Source): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Target to fake Source to fake Target) y -> x -> y
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

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }



if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    cyclegan_generator_path = sys.argv[1]
    classifier_training_data_set_path = sys.argv[2]
    classifier_test_data_set_path = sys.argv[3]
    classifier_name = sys.argv[4]
    
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    path = classifier_name + '_' + time
    os.mkdir(path)
    print("Directory '% s' created" % path)
    path = path + '/'


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
    cycle_gan_model.load_weights(cyclegan_generator_path).expect_partial()
    print("Weights loaded successfully")



    data_dir = pathlib.Path(classifier_training_data_set_path)
    train_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)

    test_data_dir = pathlib.Path(classifier_test_data_set_path)
    test_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*'), shuffle=False)

    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    print(class_names)  
    no_of_classes = len(class_names)

    list_of_files = list(data_dir.glob('*/*.png'))
    #print(list(data_dir.glob('*/*.png')))  

    image_count = len(list(data_dir.glob('*/*.png')))
    print(image_count)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())

    train_ds = train_ds.map(lambda x: preprocess_classifier_test_images(x, class_names), 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(lambda x: preprocess_classifier_test_images(x, class_names), 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = configure_for_performance_without_shuffle(train_ds, 1)
    test_ds = configure_for_performance(test_ds, 1162, 
    tf.data.experimental.cardinality(test_ds).numpy())

    print('Retrieving Original Labels')
    original_labels =  np.asarray(list(train_ds.map(lambda y, x: x)))
    
    print(original_labels.shape)
    original_labels = np.reshape(original_labels, (image_count, no_of_classes))
    print(original_labels.shape)
    save(path + 'Generated_Target_Domain_Images_Labels_' + time + '.npy', original_labels)
    original_labels_ds = tf.data.Dataset.from_tensor_slices((original_labels))
    
   
    plt.figure(figsize=(10,10))
    for i in range(10):
      X_train_10_samples, y_train_10_samples = next(iter(train_ds))
      plt.subplot(5, 2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(X_train_10_samples[i], cmap=plt.cm.gray)
      plt.xlabel(class_names[np.argmax(y_train_10_samples[i], axis=-1)])
      plt.show()
      plt.savefig(path + 'Save_Sample_Images_1')
    plt.close()
    

   
    plt.figure(figsize=(10,10))
    for i in range(10):
      X_train_10_samples, y_train_10_samples = next(iter(train_ds))
      plt.subplot(5, 2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(X_train_10_samples[i], cmap=plt.cm.gray)
      plt.xlabel(class_names[np.argmax(y_train_10_samples[i], axis=-1)])
      plt.show()
      plt.savefig(path + 'Save_Sample_Images_2')
    plt.close()


    # Generation of Target Domain Document Images
    with tf.device("cpu:0"):
        generated_target_domain_images = cycle_gan_model.gen_G.predict(train_ds)
    print('Generated target domain images shape')
    print(generated_target_domain_images.shape)
    
    save(path + 'Generated_Target_Domain_Images_' + time + '.npy', generated_target_domain_images)

    generated_target_domain_images_ds = tf.data.Dataset.from_tensor_slices((generated_target_domain_images))
    print(tf.data.experimental.cardinality(generated_target_domain_images_ds).numpy())

    final_train_ds = tf.data.Dataset.zip((generated_target_domain_images_ds, original_labels_ds))  
    print(tf.data.experimental.cardinality(final_train_ds).numpy())


    plt.figure(figsize=(10,10))
    for i in range(10):
      plt.subplot(5, 2, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(generated_target_domain_images[i], cmap=plt.cm.gray)
      plt.xlabel(class_names[np.argmax(original_labels[i], axis=-1)])
      plt.show()
      plt.savefig(path + 'Generated_Sample_Images')
    plt.close()

    # Train the Domain Adapted Realistic Document Image Classifier and 
    # Verify on Annotated Test Data.
   
    domain_adapted_documents_classifier_model = create_model(10)

    # Tensorboard Logs
    log_dir = path + "logs/fit/" + time   
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Creates a file writer for the log directory.
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
      tf.summary.image("10 training data examples", generated_target_domain_images[0:10], 
      max_outputs=25, step=0)
 
    # Prepare the plot
    figure = image_grid(generated_target_domain_images[0:10], class_names, 
    np.argmax(original_labels[0:10], axis=-1))

    # Convert to image and log
    with file_writer.as_default():
      tf.summary.image("Training data", plot_to_image(figure), step=0)

    # retrieve the test data
    X_test, y_test = next(iter(test_ds))

    # Shuffle final training dataset
    final_train_ds = final_train_ds.shuffle(tf.data.experimental.cardinality(final_train_ds).numpy())

    val_size = int(image_count * 0.2)
    final_train_ds = final_train_ds.skip(val_size)
    val_ds = final_train_ds.take(val_size)

    final_train_ds = configure_for_performance(final_train_ds, 1,
    tf.data.experimental.cardinality(final_train_ds).numpy())
    val_ds = configure_for_performance(val_ds, 1,
    tf.data.experimental.cardinality(val_ds).numpy())

    file_name = path + 'Confusion_Matrix_' + classifier_name + '_' + time
    file_name_plot = classifier_name + '_' + time

    # Example: Confusion_Matrix_CycleGAN_Generated_Data_Classifier
    epochs = 10
    history = domain_adapted_documents_classifier_model.fit(
            final_train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[tensorboard_callback,
            CustomCallback(domain_adapted_documents_classifier_model, 
            X_test,
            np.argmax(y_test, axis=-1),
            class_names,
            log_dir,
            file_writer_cm,
            file_name
            )]
            ) 
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy', fontweight='bold', fontsize='x-large')
    plt.ylabel('Accuracy', fontweight='bold', fontsize='x-large')
    plt.xlabel('Epochs', fontweight='bold', fontsize='x-large')
    plt.legend(['Train', 'Validation'], loc='upper left', prop=dict(weight='bold'))
    plt.grid(True)
    plt.show()
    plt.savefig(path + file_name_plot + '_Accuracy', dpi=300)
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss', fontweight='bold', fontsize='x-large')
    plt.ylabel('Loss', fontweight='bold', fontsize='x-large')
    plt.xlabel('Epochs', fontweight='bold', fontsize='x-large')
    plt.legend(['Train', 'Validation'], loc='upper left', prop=dict(weight='bold'))
    plt.grid(True)
    plt.show()
    plt.savefig(path + file_name_plot + '_Loss', dpi=300)
    plt.close()

    print('Training Finished...')
     # serialize weights to HDF5
    domain_adapted_documents_classifier_model.save(path + classifier_name + '_' + time + '_model.h5')
    
    domain_adapted_documents_classifier_logs =  open(path + classifier_name + '_logs'
    '_' + time + '.txt', 'a')
    
    print('Saved model to disk ' + classifier_name + '_' + time + '_model.h5', 
    file=domain_adapted_documents_classifier_logs)


    print('translated target domain images and labels shape', file=domain_adapted_documents_classifier_logs)
    print(generated_target_domain_images.shape, file=domain_adapted_documents_classifier_logs) 
    print(original_labels.shape, file=domain_adapted_documents_classifier_logs)


    y_test_pred = np.argmax(domain_adapted_documents_classifier_model.predict(X_test), axis=-1)
    y_test_real = np.argmax(y_test, axis=-1)

    results = domain_adapted_documents_classifier_model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_real, y_test_pred, target_names=class_names, zero_division=1), 
    file=domain_adapted_documents_classifier_logs)
    
    print("test loss, test acc:", results, file=domain_adapted_documents_classifier_logs)

  
    domain_adapted_documents_classifier_logs.close()