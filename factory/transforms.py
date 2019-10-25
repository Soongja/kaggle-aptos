import os
import random
import numpy as np
import torch
import cv2
from albumentations import (
    OneOf, Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur,
    CLAHE, IAASharpen, GaussNoise, IAAAdditiveGaussianNoise,
    HueSaturationValue, RGBShift, ChannelShuffle,
    Rotate, Resize, RandomCrop, Flip, RandomRotate90,
    ToGray, RandomSizedCrop)


class Normalize:
    """
    normalize data to -1 ~ 1
    """
    def __call__(self, data):
        smooth = 1e-6

        # data = (data - np.min(data) + smooth) / (np.max(data) - np.min(data) + smooth)
        data = data / 255.0
        data = data * 2 - 1

        return data


class ToTensor:
    """
    convert ndarrays to Tensors.
    """
    def __call__(self, data):
        data = np.transpose(data, (2, 0, 1))
        data = torch.from_numpy(data).float()
        return data


def weak_aug(p=1.0):
    return Compose([
        # 순서 고민한거임
        # HorizontalFlip(p=0.5),
        Rotate(limit=360, border_mode=0, p=0.7),
        RandomSizedCrop(min_max_height=(int(512*0.8), 512), height=512, width=512, p=0.7),
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    ], p=p)


class Albu():
    def __call__(self, image):
        augmentation = weak_aug()

        data = {"image": image}
        augmented = augmentation(**data)

        return augmented["image"]


class CV2_Resize():
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, image):
        image = cv2.resize(image, (self.w, self.h))

        return image


if __name__=="__main__":
    # dir = '../data/test_images_ben_sigma30_512'
    # fnames = [os.path.join(dir, f) for f in os.listdir(dir)]
    img = cv2.imread('../data/train_images_ben_sigma30_512/000c1434d8d7.png', 1)

    # for f in fnames:
    #     img = cv2.imread(f, 1)
    for i in range(100):
        data = {"image": img}
        aug = Compose([
            # 순서 고민한거임
            # HorizontalFlip(p=0.5),
            # RandomSizedCrop(min_max_height=(int(512 * 0.75), 512), height=512, width=512, p=0.7),
            # Rotate(limit=360, border_mode=0, p=0.7),
            RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.5)
            # CLAHE(clip_limit=2.0, p=1.0),
    ], p=1.0)
        augmented = aug(**data)

        out_img = augmented["image"]
        # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('img', img)
        cv2.imshow('out', out_img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
