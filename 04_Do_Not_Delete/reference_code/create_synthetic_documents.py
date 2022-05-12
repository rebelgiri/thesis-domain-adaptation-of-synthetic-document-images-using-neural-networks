from pathlib import Path
from aiseg import modify
import sys
import os


def create_synthetic_documents(path_templates_with_annotations, path_crops,
                            visual_object_classes_numeric,
                            visual_object_classes_alpha, path_output, template_folder_name):
                
    template_directory_path = Path(path_templates_with_annotations)
    output_directory_path = Path(path_output)
    
    # crop_images_directory_path = Path('D:/Projects/MNISTExtractor/Code/mnist_png/data/output/training/0')
    
    crop_images_directory_path = Path(path_crops)
    visual_object_classes = [visual_object_classes_numeric, visual_object_classes_alpha]
    number_of_samples_per_template = 10000

    # print(path.exists('../../dataset/templates/'))
    # print(path.exists('../../dataset/output/'))

    print('Creating Sythetic Document Images for ' + template_folder_name)
    modify.create_synthetic_samples(template_directory_path, output_directory_path,
                                    crop_images_directory_path, visual_object_classes,
                                    number_of_samples_per_template)

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    templates_with_annotations_path = sys.argv[1]
    crops_path = sys.argv[2]
    visual_object_classes_numeric_1 = sys.argv[3]
    visual_object_classes_alpha_2 = sys.argv[4]
    output_path = sys.argv[5]

    
    for template_folder_name in os.listdir(templates_with_annotations_path):
        if(True == os.path.exists(output_path +  'synthetic_document_images_classifier/' + template_folder_name)):
            continue
        create_synthetic_documents(templates_with_annotations_path + template_folder_name, crops_path, 
                visual_object_classes_numeric_1,
                visual_object_classes_alpha_2,
                output_path +  'synthetic_document_images_classifier/' + template_folder_name, template_folder_name )
