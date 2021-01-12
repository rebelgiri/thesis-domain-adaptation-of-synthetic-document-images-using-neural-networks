import dataclasses
import json
from pathlib import Path
import numpy as np
import PIL.Image
import os.path
from os import path
from aiseg import modify


def create_synthetic_documents():
    template_directory_path = Path('../../dataset/templates/')
    output_directory_path = Path('../../dataset/output/')
    # crop_images_directory_path = Path('D:/Projects/MNISTExtractor/Code/mnist_png/data/output/training/0')
    crop_images_directory_path = Path('../../dataset/crops/')
    visual_object_classes = ['http://ai4bd.com/resource/cdm/juzo/numeric', 'http://ai4bd.com/resource/cdm/juzo/alpha']
    number_of_samples_per_template = 1

    print(path.exists('../../dataset/templates/'))
    print(path.exists('../../dataset/output/'))

    modify.create_synthetic_samples(template_directory_path, output_directory_path,
                                    crop_images_directory_path, visual_object_classes,
                                    number_of_samples_per_template)


def main():
    create_synthetic_documents()


if __name__ == "__main__":
    main()
