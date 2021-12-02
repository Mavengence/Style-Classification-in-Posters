import kornia.augmentation as K
import torch.nn as nn
import torch
import numpy as np


transform = nn.Sequential(
    K.RandomAffine(360),
    K.ColorJitter(0.2, 0.3, 0.2, 0.3),
    #K.CenterCrop(int(np.random.randint(180, 224, size=(1))[0])),
    K.RandomHorizontalFlip(0.5),
    K.RandomVerticalFlip(0.5),
    )


def augment_images(images):
    return transform(images)