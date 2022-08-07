import sys
# if you have some problem with packages edit and uncomment these methods.
# sys.path.append('c:/users/USR_NAME/appdata/local/packages/pythonsoftwarefoundation.YOUR_PYTHON_DIR/localcache/local-packages/python_YOUR_DIR/site-packages')
# sys.path.insert(0, "..")

import os
import cv2
import albumentations as A
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NOF_AUG = 8
IMG_CROP_RATIO = 4

index = 0
imgSrcDir = sys.argv[1]
lblSrcDir = sys.argv[2]
imgFiles = os.listdir(imgSrcDir)
lblFiles = os.listdir(lblSrcDir)

dirName = "aug_out"
dirImageName = "images"
dirLabelName = "labels"
directory = dirName
dirImg = ""
dirLabel = ""
path = os.path.join("./", directory)
dirCnt = 1
#print(imgFiles)
#print(lblFiles)
while True:
    if os.path.exists(path) == False:
        path = os.path.join("./", directory)
        os.makedirs(path)
        dirImg = directory + "/" + dirImageName
        path = os.path.join("./", dirImg)
        os.makedirs(path)
        dirLabel = directory + "/" + dirLabelName
        path = os.path.join("./", dirLabel)
        os.makedirs(path)
        break
    else:
        directory = dirName + str(dirCnt)
        path = os.path.join("./", directory)
        dirCnt = dirCnt + 1

for imgFileName in imgFiles:
    inImgName = imgFileName[:-4] #      removing .jpg format.
    inLblName = lblFiles[index][:-4] #  removing .txt format.
    if(imgFileName == "desktop.ini" or inLblName == "desktop.ini"):
        index = index + 1
        continue

    def yoloToPascalVoc(xCenter, yCenter, w, h,  imageW, imageH):
        w = w * imageW
        h = h * imageH
        x1 = ((2 * xCenter * imageW) - w)/2
        y1 = ((2 * yCenter * imageH) - h)/2
        x2 = x1 + w
        y2 = y1 + h
        return [x1, y1, x2, y2]

    image = Image.open(imgSrcDir + "/" + inImgName + ".jpg")
    with open(lblSrcDir + "/" + inLblName + ".txt") as f:
        lines = f.readlines()

    class_labels = []
    classLabelList = []
    bboxes = []
    bboxesList = []

    for i in range(len(lines)):
        splitted = lines[i].split()
        class_labels.append(int(splitted[0]))
        values = []
        for j in range(len(splitted) - 1):
            values.append(float(splitted[j + 1]))
        bboxes.append(values)

    h = image.height
    w = image.width

    transform = A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(int(h - h/IMG_CROP_RATIO), int(w - w/IMG_CROP_RATIO)),
            A.RandomRain(),
            A.RandomBrightnessContrast(),
            A.ChannelShuffle(),
            A.ColorJitter(),
            A.Rotate(limit = 40, p = 0.9),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.1),
            A.OneOf([
                A.Blur(blur_limit = 3, p = 0.5),
                A.ColorJitter(p = 0.5)], p = 1.0),
            ], bbox_params = A.BboxParams(format = 'yolo', label_fields = ['class_labels'])
        )
    imageList = [image]
    image = np.array(image)
    if(len(image.shape) < 3): # checking nof color channel.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in range(NOF_AUG):
        augmentations = transform(image = image, bboxes = bboxes, class_labels = class_labels)
        augmented_img = augmentations['image']
        bboxesList.append(augmentations['bboxes'])
        classLabelList.append(augmentations['class_labels'])
        imageList.append(augmented_img)

    imgCnt = 0
    for i in range(len(imageList) - 1):
        cv2.imwrite(dirImg + "/" + inImgName + "_o" + str(imgCnt + 1) + ".jpg",imageList[imgCnt + 1])
        imgCnt = imgCnt + 1
    lblCnt = 0

    for i in range(len(bboxesList)):
        fLbl = open(dirLabel + "/" + inLblName + "_o" + str(lblCnt + 1) + ".txt", "a")
        for j in range(len(bboxesList[i])):
            temp = str(classLabelList[i][j]) + " "
            for k in range(len(bboxesList[i][j])):
                temp = temp + str(bboxesList[i][j][k])[:8] + " "
            temp = temp[:-1]
            temp = temp + "\n"
            fLbl.write(temp)
        fLbl.close()
        lblCnt = lblCnt + 1
    index = index + 1
    print(inImgName + " augmentation completed! Total completed augmentation: " + str(index))
