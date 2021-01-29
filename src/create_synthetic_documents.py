from pathlib import Path
from aiseg import modify
import sys


def create_synthetic_documents(path_templates_with_annotations, path_crops,
                            visual_object_classes_numeric,
                            visual_object_classes_alpha, path_output):
                
    template_directory_path = Path(path_templates_with_annotations)
    output_directory_path = Path(path_output + '/synthetic_document_images/')
    
    # crop_images_directory_path = Path('D:/Projects/MNISTExtractor/Code/mnist_png/data/output/training/0')
    
    crop_images_directory_path = Path(path_crops)
    visual_object_classes = [visual_object_classes_numeric, visual_object_classes_alpha]
    number_of_samples_per_template = 11

    # print(path.exists('../../dataset/templates/'))
    # print(path.exists('../../dataset/output/'))

    modify.create_synthetic_samples(template_directory_path, output_directory_path,
                                    crop_images_directory_path, visual_object_classes,
                                    number_of_samples_per_template)


# def main():
#    create_synthetic_documents()


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    templates_with_annotations_path = sys.argv[1]
    crops_path = sys.argv[2]
    visual_object_classes_numeric_1 = sys.argv[3]
    visual_object_classes_alpha_2 = sys.argv[4]
    output_path = sys.argv[5]
    create_synthetic_documents(templates_with_annotations_path, crops_path, 
                visual_object_classes_numeric_1,
                visual_object_classes_alpha_2,
                output_path)
