# Image_Augmentation_for_YOLO
You can easily expand your data set with this script.

# How to use?

Write to this command line to your command prompt: 

``` python albumentations_yolo_script.py your_image_directory your_label_bbox_directory ```

After this command execution, script will create "aug_out" folder and add the images and to bbox labels to this folder.
You can change augmentation number of each image with "NOF_AUG" macro. 

# Output Example
Here the bike image has exposed 8 image augmentation with %25 crop rate.

Original image taken by Pavan Sanagapati's data set from Kaggle repo.

Data set link: https://www.kaggle.com/datasets/pavansanagapati/images-dataset

![output_example](https://user-images.githubusercontent.com/45585791/183306763-5faa24c1-ac38-4234-8f8f-4a59d37e9ec6.JPG)