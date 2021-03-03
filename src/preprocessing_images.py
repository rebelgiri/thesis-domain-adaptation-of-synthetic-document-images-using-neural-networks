

import tensorflow as tf

def normalize_img(img):
    # Map values in the range [-1, 1]
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def preprocess_train_image(image_path):
    img = tf.io.read_file(image_path)
    # Convert the image in grayscale
    img = tf.image.decode_png(img, channels=1)
    # Resize the image [[256, 256]]
    img = tf.image.resize(img, [256, 256])
    img = normalize_img(img)
    label = 0
    return img, label
