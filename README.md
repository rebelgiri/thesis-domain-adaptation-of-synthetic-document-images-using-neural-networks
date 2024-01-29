
# Domain adaptation of synthetic document images using neural networks


## Abstract

Neural networks have improved significantly in past decades. They are competent
to solve complex problems in the field of deep learning and they are capable to
manage a large amount of complex data like images, videos and sound. However, the
training of neural networks requires a significantly large amount of annotated data,
which is not always possible. Machine learning engineers inevitably have to generate
synthetic data. Although, the neural networks trained on synthetic data will not able
to generalize well on real data. In recent years, an effective technique named domain
adaptation has evolved, to address the problem of scarcity of annotated data. The domain
adaptation technique can transform data from the source domain to the target
domain. For example, domain adaptation techniques like image-to-image translation
can be used to transform images of zebras into images of horses and vice-versa.
This thesis proposes an image-to-image translation application that aims to reduce
the domain gap between synthetic and real data distribution using Cycle-Consistent
Adversarial Networks (CycleGANs). The proposed application is used to transform
synthetic document images into realistic document images, to overcome the scarcity
of annotated real document images. In addition, these generated realistic document
images are used to train a classifier to classify similar unlabeled real document images,
thereby accelerating the process of labeling images in an unsupervised and automated
manner. Experimental results show the generated realistic document images are qualitatively
convincing and need improvement quantitatively to match the real data distribution
significantly. Such preliminary results show that CycleGAN can solve the
problem of data scarcity by generating high-quality images in the target domain. The
purpose of this thesis is limited to improving the classification of real document images.
Once the rich and sufficient data is generated in the target domain, the performance
of the real document image classifier eventually can be improved. This thesis is
limited to the study of unpaired image-to-image translation method CycleGAN. The
remaining methods and comparisons with them are left for future work. In the future,
CycleGAN can be used to generate high-quality realistic images in many tasks,
such as handwriting recognition, image classification, image segmentation and object
detection.

## Training CycleGAN to transform Synthetic Images into Realistic Images

- Use script **cyclegan_keras_example.py** to start training.
- Inside the script two variables **synthetic_document_images_path** and **real_document_images_path** values needs to be modified.
- Assign the correct path for synthetic document images and real document images.
- You can also assign a value to a variable **synthetic_document_images_path_test**, once the training is finished (after 20 epochs), the last saved checkpoint is loaded and the synthetic images from the testing dataset are transformed into realistic document images, so the developer can have a qualitative overview of images produced by the generator.

## Problem

![image](https://user-images.githubusercontent.com/18268525/166524949-a84a1dda-da3c-4ab8-b02f-936a1f1f78ac.png)

## Proposed Solution

![image](https://user-images.githubusercontent.com/18268525/166525063-031d4044-19c3-41cf-a756-bd0ae05ded99.png)

## CycleGAN Architecture

### Discriminator Architecture

![image](https://user-images.githubusercontent.com/18268525/166551564-192db3c7-7b93-48fd-82c2-48d0a9fd05c8.png)

### Generator Architecture

![image](https://user-images.githubusercontent.com/18268525/166551703-954bf35c-546e-4879-bd97-bba9c7f4e0f7.png)

## Loss Functions

### Forward Cycle-Consistency Loss

![image](https://user-images.githubusercontent.com/18268525/166524802-f617f1b0-bd4a-4201-8c3d-dedaa7a08801.png)

### Backward Cycle-Consistency Loss

![image](https://user-images.githubusercontent.com/18268525/166525635-8c5addd6-83cf-4938-8b78-1078bca690e9.png)

### Generator Loss

![image](https://user-images.githubusercontent.com/18268525/166548185-90965f8d-3a1f-4122-b7f6-9f3b5946e76c.png)

### Discriminator Loss

![image](https://user-images.githubusercontent.com/18268525/166549827-1b64c00e-4aeb-44cd-9d57-88bee8bb13e9.png)



