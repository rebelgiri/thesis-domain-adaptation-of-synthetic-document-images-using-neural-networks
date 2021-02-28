from matplotlib.pyplot import imsave, imshow
from numpy.lib.function_base import append
from synthetic_documents_classifier import *
from datetime import datetime
# from data_set_loader_pytorch import *
from data_set_loader_keras import *
import sys
import tensorflow as tf
import io
import itertools
import sklearn.metrics

class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, test_images, test_labels, class_names, log_dir, file_writer_cm):
        self.model = model
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_names = class_names
        self.log_dir = log_dir
        self.file_writer_cm = file_writer_cm

    def on_epoch_end(self, epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.test_images)
        test_pred = np.argmax(test_pred_raw, axis=1)
        
        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(self.test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
          tf.summary.image("Confusion Matrix", cm_image, step=epoch)
  

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def image_grid(train_images, class_names, train_labels):
  """Return a 2x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(20,20))
  for i in range(10):
    # Start next subplot.
    plt.subplot(2, 5, i + 1, title=class_names[train_labels[i]])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)

  return figure


def normalize(image_tensor):
  image_tensor = (image_tensor / 127.5) - 1
  return image_tensor

if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    classifier_training_data_set_path = sys.argv[1]
    classifier_validation_data_set_path = sys.argv[2] 
    classifier_test_data_set_path = sys.argv[3]

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # create generator
    datagen = ImageDataGenerator(preprocessing_function=normalize)

    # prepare an iterators for training dataset
    train_it = datagen.flow_from_directory(classifier_training_data_set_path, 
      color_mode='grayscale',
      shuffle=True,
      batch_size=10,
      interpolation='bilinear')

    # batchX, batchy = train_it.next()
    # print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    # prepare an iterators for validation dataset
    val_it = datagen.flow_from_directory(classifier_validation_data_set_path, 
      color_mode='grayscale',
      shuffle=True,
      batch_size=10,
      interpolation='bilinear')

    # batchX, batchy = val_it.next()
    # print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    # prepare an iterators for testing dataset
    test_it = datagen.flow_from_directory(classifier_test_data_set_path, 
      color_mode='grayscale',
      shuffle=True,
      batch_size=1162,
      interpolation='bilinear')
    
    classes = list()
    for key in train_it.class_indices:
        classes.append(key)

    # print(classes)
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = "logs/fit/" + time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Creates a file writer for the log directory.
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
      batchX, batchY = train_it.next()
      tf.summary.image("10 training data examples", batchX, max_outputs=25, step=0)
 
    # Prepare the plot
    figure = image_grid(batchX, classes, np.argmax(batchY, axis=-1))
    # Convert to image and log
    with file_writer.as_default():
      tf.summary.image("Training data", plot_to_image(figure), step=0)

    # Define the per-epoch callback.
    # cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    # create classifier model
    type_of_the_classifier = 'synthetic_documents_classifier'
    synthetic_documents_classifier_model = create_model(10)     

    # retrieve the test data
    X_test, y_test = test_it.next()

    synthetic_documents_classifier_model.fit(
            train_it,
            steps_per_epoch=10000, # 10000
            epochs=5,
            validation_data=val_it,
            validation_steps=2000, # 2000
            callbacks=[tensorboard_callback, CustomCallback(synthetic_documents_classifier_model, 
            X_test,
            np.argmax(y_test, axis=-1),
            classes,
            log_dir,
            file_writer_cm
            )])
          
    print('Training Finished...')
    # serialize weights to HDF5
    synthetic_documents_classifier_model.save(type_of_the_classifier + '_' + time + '_model.h5')
    
    synthetic_documents_classifier_logs =  open('synthetic_documents_classifier_logs' + 
    '_' + time + '.txt', 'a')

    print('Saved model to disk ' + type_of_the_classifier + '_' + time + '_model.h5', 
    file=synthetic_documents_classifier_logs)

    y_test_pred = np.argmax(synthetic_documents_classifier_model.predict(X_test), axis=-1)
    y_test_real = np.argmax(y_test, axis=-1)

    results = synthetic_documents_classifier_model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_real, y_test_pred, target_names=classes, zero_division=1), file=synthetic_documents_classifier_logs)
    print("test loss, test acc:", results, file=synthetic_documents_classifier_logs)

    # serialize weights to HDF5
    synthetic_documents_classifier_model.save(type_of_the_classifier + '_' + time + '_model.h5')

    synthetic_documents_classifier_logs.close()

   