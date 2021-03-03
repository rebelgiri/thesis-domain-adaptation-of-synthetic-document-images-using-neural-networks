from pathlib import Path
from aiseg import modify
import sys
import os


def create_synthetic_documents(path_templates_with_annotations, crop_images_alphanumeric,
                            crop_images_numeric, path_output, template_folder_name_):
               
    crop_images_alphanumeric = Path(crop_images_alphanumeric)
    crop_images_numeric = Path(crop_images_numeric)
    crop_images_directory_list = [crop_images_alphanumeric, crop_images_numeric]
    crop_images_directory_labels = ["alpha", "numeric"]
    number_of_samples_per_template = 10000
    templates_directory_path = Path(path_templates_with_annotations)
    output_directory_path = Path(path_output)
   
    print('Creating Sythetic Document Images for ' + template_folder_name_)
    modify.create_synthetic_samples(
        templates_directory_path=templates_directory_path,
        output_directory_path=output_directory_path,
        crop_images_directory_list=crop_images_directory_list,
        crop_images_directory_labels=crop_images_directory_labels,
        number_of_samples_per_template=number_of_samples_per_template
)

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    templates_with_annotations_path = sys.argv[1]
    crop_images_alphanumeric_ = sys.argv[2]
    crop_images_numeric_ = sys.argv[3]
    output_path = sys.argv[4]

    for template_folder_name in os.listdir(templates_with_annotations_path):
        if os.path.exists(output_path +  'synthetic_document_images_classifier/' + template_folder_name):
            continue
        create_synthetic_documents(templates_with_annotations_path + template_folder_name,
                crop_images_alphanumeric_,
                crop_images_numeric_,
                output_path +  'synthetic_document_images_classifier/' + template_folder_name,
                template_folder_name )
