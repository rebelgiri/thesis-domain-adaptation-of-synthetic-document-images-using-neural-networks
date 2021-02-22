
from matplotlib.pyplot import imsave, imshow
from numpy.lib.function_base import append
from synthetic_documents_classifier import *
from datetime import datetime
# from data_set_loader_pytorch import *
from data_set_loader_keras import *
import sys
import tensorflow as tf


def normalize(image):
  image = (image / 127.5) - 1
  return image

if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    classifier_training_data_set_path = sys.argv[1]
    classifier_test_data_set_path = sys.argv[2]
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # create generator
    datagen = ImageDataGenerator(preprocessing_function=normalize,
    validation_split=0.2)

    # prepare an iterators for training dataset
    train_it = datagen.flow_from_directory(classifier_training_data_set_path, 
    color_mode='grayscale',
    shuffle=True,
    batch_size=10,
    interpolation='bilinear',
    subset='training')

    # batchX, batchy = train_it.next()
    # print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    # prepare an iterators for validation dataset
    val_it = datagen.flow_from_directory(classifier_training_data_set_path, 
    color_mode='grayscale',
    shuffle=True,
    batch_size=10,
    interpolation='bilinear',
    subset='validation')

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

    # create classifier model
    type_of_the_classifier = 'synthetic_documents_classifier'
    synthetic_documents_classifier_model = create_model(10)     

 
    history = list()
    
    history.append(synthetic_documents_classifier_model.fit(
            train_it,
            steps_per_epoch=10000,
            epochs=10,
            validation_data=val_it,
            validation_steps=2000,
            callbacks=[tensorboard_callback]))
          
    print('Training Finished...')
    
    # Save the results
    length = len(history)
    h = np.zeros((length, 5), dtype=np.float32)

    for i in range(length):
        h[i, 0] = i
        h[i, 1] = np.array(history[i].history['accuracy'])
        h[i, 2] = np.array(history[i].history['loss'])
        h[i, 3] = np.array(history[i].history['val_accuracy'])
        h[i, 4] = np.array(history[i].history['val_loss'])

    plt.plot(h[:, 1] * 100, '--')
    plt.plot(h[:, 2] * 100, '-.')
    plt.plot(h[:, 3] * 100, ':')
    plt.plot(h[:, 4] * 100, '-')

    plt.legend(['acc', 'loss', 'val_acc', 'val_loss'], loc='lower right')
    plt.axis([0, length, 0, 100])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Error in percentage')
    plt.grid('on')

    plt.show()
    plt.savefig(type_of_the_classifier + '_' + time + '_model.png')
    plt.close()

    X_test, y_test = test_it.next()

    y_test_pred = np.argmax(synthetic_documents_classifier_model.predict(X_test), axis=-1)
    y_test_real = np.argmax(y_test, axis=-1)

    synthetic_documents_classifier_logs =  open('synthetic_documents_classifier_logs' + 
    '_' + time + '.txt', 'a')
    results = synthetic_documents_classifier_model.evaluate(X_test, y_test, verbose=2)
    print(classification_report(y_test_real, y_test_pred, target_names=classes, zero_division=1), file=synthetic_documents_classifier_logs)
    print("test loss, test acc:", results, file=synthetic_documents_classifier_logs)

    # serialize weights to HDF5
    synthetic_documents_classifier_model.save(type_of_the_classifier + '_' + time + '_model.h5')

    print('Saved model to disk ' + type_of_the_classifier + '_' + time + '_model.h5', 
    file=synthetic_documents_classifier_logs)

    synthetic_documents_classifier_logs.close()

   