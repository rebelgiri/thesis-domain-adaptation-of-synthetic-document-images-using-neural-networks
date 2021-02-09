
from cyclegan_resnet import *
from synthetic_documents_classifier import *
from datetime import datetime

# Algorithm Steps:
# Train the CycleGAN.
# Train the Synthetic Document Image Classifier and Verify on Annotated Test Data.


if __name__ == "__main__":

    print(f"Arguments count: {len(sys.argv)}")
    synthetic_document_images_path = sys.argv[1]
    real_document_images_path = sys.argv[2]
    classifier_training_data_set_path = sys.argv[3]
    classifier_test_data_set_path = sys.argv[4]

    cyclegan_training_logs = open('cyclegan_training_logs.txt', 'a')
    print(datetime.now(), file=cyclegan_training_logs)

    # Train the CycleGAN.
    
    # input shape
    image_shape = (256, 256, 1)

    # generator: A -> B
    g_model_AtoB = define_generator(image_shape)
    # g_model_AtoB.summary()
    # plot_model(g_model_AtoB, to_file='g_model_AtoB_plot.png', show_shapes=True, show_layer_names=True)

    # generator: B -> A
    g_model_BtoA = define_generator(image_shape)
    # g_model_BtoA.summary()
    # plot_model(g_model_BtoA, to_file='g_model_BtoA_plot.png', show_shapes=True, show_layer_names=True)

    # discriminator: A -> [real/fake]
    d_model_A = define_discriminator(image_shape)
    # d_model_A.summary()
    # plot_model(d_model_A, to_file='d_model_A_plot.png', show_shapes=True, show_layer_names=True)

    # discriminator: B -> [real/fake]
    d_model_B = define_discriminator(image_shape)
    # d_model_B.summary()
    # plot_model(d_model_B, to_file='d_model_B_plot.png', show_shapes=True, show_layer_names=True)

    # composite: A -> B -> [real/fake, A]
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    # c_model_AtoB.summary()
    # plot_model(c_model_AtoB, to_file='c_model_AtoB_plot.png', show_shapes=True, show_layer_names=True)

    # composite: B -> A -> [real/fake, B]
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
    # c_model_BtoA.summary()
    # plot_model(c_model_BtoA, to_file='c_model_BtoA_plot.png', show_shapes=True, show_layer_names=True)

    data_set_cyclegan = cyclegan_load_data_set(synthetic_document_images_path, real_document_images_path)
    train_cyclegan(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, 
        c_model_AtoB, c_model_BtoA, data_set_cyclegan, cyclegan_training_logs)

    # Train the Synthetic Document Image Classifier and Verify in Annotated Test Data.
    classifier_training_data_set, list_of_name_of_template = classifier_load_data_set(classifier_training_data_set_path)
    classifier_test_data_set = classifier_load_data_set(classifier_test_data_set_path)
    type_of_the_classifier = 'synthetic_documents_classifier'
    synthetic_documents_classifier_model = create_model(10)
    # synthetic_documents_classifier_model.summary()
    # plot_model(synthetic_documents_classifier_model, to_file=type_of_the_classifier + '.png', show_shapes=True, show_layer_names=True)
    
    start_training_classifier(synthetic_documents_classifier_model, 
        classifier_training_data_set, classifier_test_data_set, type_of_the_classifier,
        list_of_name_of_template)

    print('---------------------------------------------------------------------------------', 
    file=cyclegan_training_logs)
    
    cyclegan_training_logs.close()    

    





